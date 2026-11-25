//! Minimal glTF 2.0 / VRM 1.0 writer.
//!
//! The scene produced by the editor has already been edited in place (morphs,
//! body and face shapes are baked into vertex positions), so the exporter only
//! has to serialise the current state: meshes, materials, textures, skins and
//! the `VRMC_vrm` extension with human bone / expression bindings.

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

use crate::{AlphaMode, Scene};

const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

struct Writer {
    bytes: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            views: Vec::new(),
            accessors: Vec::new(),
        }
    }

    fn view(&mut self, data: &[u8], target: Option<u32>) -> usize {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        let offset = self.bytes.len();
        self.bytes.extend_from_slice(data);
        let mut v = Map::new();
        v.insert("buffer".into(), json!(0));
        v.insert("byteOffset".into(), json!(offset));
        v.insert("byteLength".into(), json!(data.len()));
        if let Some(t) = target {
            v.insert("target".into(), json!(t));
        }
        self.views.push(Value::Object(v));
        self.views.len() - 1
    }

    fn accessor(
        &mut self,
        view: usize,
        component: u32,
        ty: &str,
        count: usize,
        min: Option<Vec<f32>>,
        max: Option<Vec<f32>>,
    ) -> usize {
        let mut a = Map::new();
        a.insert("bufferView".into(), json!(view));
        a.insert("componentType".into(), json!(component));
        a.insert("count".into(), json!(count));
        a.insert("type".into(), json!(ty));
        if let Some(v) = min {
            a.insert("min".into(), json!(v));
        }
        if let Some(v) = max {
            a.insert("max".into(), json!(v));
        }
        self.accessors.push(Value::Object(a));
        self.accessors.len() - 1
    }
}

fn human_bones(scene: &Scene) -> Value {
    let mut pairs: Vec<(String, String)> = [
        ("hips", "J_Bip_C_Hips"),
        ("spine", "J_Bip_C_Spine"),
        ("chest", "J_Bip_C_Chest"),
        ("upperChest", "J_Bip_C_UpperChest"),
        ("neck", "J_Bip_C_Neck"),
        ("head", "J_Bip_C_Head"),
        ("leftEye", "J_Adj_L_FaceEye"),
        ("rightEye", "J_Adj_R_FaceEye"),
        ("jaw", "J_Adj_C_Jaw"),
    ]
    .into_iter()
    .map(|(a, b)| (a.into(), b.into()))
    .collect();
    for (side, prefix) in [("left", "L"), ("right", "R")] {
        for (bone, node) in [
            ("Shoulder", "Shoulder"),
            ("UpperArm", "UpperArm"),
            ("LowerArm", "LowerArm"),
            ("Hand", "Hand"),
            ("UpperLeg", "UpperLeg"),
            ("LowerLeg", "LowerLeg"),
            ("Foot", "Foot"),
            ("Toes", "Toes"),
        ] {
            pairs.push((format!("{side}{bone}"), format!("J_Bip_{prefix}_{node}")));
        }
        for (finger, parts) in [
            ("Thumb", ["Metacarpal", "Proximal", "Distal"]),
            ("Index", ["Proximal", "Intermediate", "Distal"]),
            ("Middle", ["Proximal", "Intermediate", "Distal"]),
            ("Ring", ["Proximal", "Intermediate", "Distal"]),
            ("Little", ["Proximal", "Intermediate", "Distal"]),
        ] {
            for (i, part) in parts.iter().enumerate() {
                pairs.push((
                    format!("{side}{finger}{part}"),
                    format!("J_Bip_{prefix}_{finger}{}", i + 1),
                ));
            }
        }
    }
    let mut out = Map::new();
    for (bone, node_name) in pairs {
        if let Some(node) = scene.nodes.iter().position(|n| n.name == node_name) {
            out.insert(bone, json!({ "node": node }));
        }
    }
    Value::Object(out)
}

fn png(texture: &crate::Texture) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let img = image::RgbaImage::from_raw(texture.width, texture.height, texture.rgba.clone())
        .context("texture size does not match pixel data")?;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(out)
}

/// Write `scene` as a VRM 1.0 GLB. `expressions` maps preset names to
/// `(node, morph target index)` binds, as produced by `expression_binds`.
pub fn export_vrm(
    scene: &Scene,
    expressions: &BTreeMap<String, Vec<(usize, usize)>>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut w = Writer::new();

    let mut images = Vec::new();
    let mut textures = Vec::new();
    for tex in &scene.textures {
        let data = png(tex)?;
        let view = w.view(&data, None);
        images.push(json!({ "bufferView": view, "mimeType": "image/png" }));
        textures.push(json!({ "sampler": 0, "source": images.len() - 1 }));
    }

    let mut materials = Vec::new();
    let mut extensions_used = vec!["VRMC_vrm".to_string()];
    for m in &scene.materials {
        let mut pbr = Map::new();
        pbr.insert("baseColorFactor".into(), json!(m.base_color));
        if let Some(t) = m.base_color_tex {
            pbr.insert("baseColorTexture".into(), json!({ "index": t }));
        }
        pbr.insert("metallicFactor".into(), json!(m.metallic));
        pbr.insert("roughnessFactor".into(), json!(m.roughness));
        let mut mat = Map::new();
        mat.insert("name".into(), json!(m.name));
        mat.insert("pbrMetallicRoughness".into(), Value::Object(pbr));
        mat.insert("emissiveFactor".into(), json!(m.emissive));
        mat.insert("doubleSided".into(), json!(m.double_sided));
        mat.insert(
            "alphaMode".into(),
            json!(match m.alpha_mode {
                AlphaMode::Opaque => "OPAQUE",
                AlphaMode::Mask => "MASK",
                AlphaMode::Blend => "BLEND",
            }),
        );
        mat.insert("alphaCutoff".into(), json!(m.alpha_cutoff));
        if let Some(t) = m.normal_tex {
            mat.insert("normalTexture".into(), json!({ "index": t }));
        }
        if m.mtoon.enabled {
            if !extensions_used.iter().any(|e| e == "VRMC_materials_mtoon") {
                extensions_used.push("VRMC_materials_mtoon".into());
            }
            mat.insert(
                "extensions".into(),
                json!({ "VRMC_materials_mtoon": {
                    "specVersion": "1.0",
                    "shadeColorFactor": m.mtoon.shade_color,
                    "shadingToonyFactor": m.mtoon.shading_toony,
                    "shadingShiftFactor": m.mtoon.shading_shift,
                    "rimColorFactor": m.mtoon.rim_color,
                    "rimLightingMixFactor": m.mtoon.rim_lighting_mix,
                    "outlineWidthFactor": m.mtoon.outline_width,
                    "outlineColorFactor": m.mtoon.outline_color,
                }}),
            );
        }
        materials.push(Value::Object(mat));
    }

    let mut skins = Vec::new();
    for skin in &scene.skins {
        let mats: Vec<[f32; 16]> = skin
            .inverse_bind
            .iter()
            .map(|m| m.to_cols_array())
            .collect();
        let flat: Vec<f32> = mats.iter().flat_map(|m| m.iter().copied()).collect();
        let view = w.view(bytemuck::cast_slice(&flat), Some(ARRAY_BUFFER));
        let acc = w.accessor(view, 5126, "MAT4", mats.len(), None, None);
        skins.push(json!({ "joints": skin.joints, "inverseBindMatrices": acc }));
    }

    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut loose: Vec<usize> = Vec::new();
    for (i, p) in scene.primitives.iter().enumerate() {
        match p.node {
            Some(n) if n < scene.nodes.len() => groups.entry(n).or_default().push(i),
            _ => loose.push(i),
        }
    }

    let mut meshes = Vec::new();
    let mut mesh_of_node: BTreeMap<usize, usize> = BTreeMap::new();
    for (&node, prims) in &groups {
        mesh_of_node.insert(node, meshes.len());
        meshes.push(build_mesh(&mut w, scene, prims));
    }
    for (i, p) in loose.iter().enumerate() {
        let mesh = meshes.len();
        meshes.push(build_mesh(&mut w, scene, std::slice::from_ref(p)));
        mesh_of_node.insert(scene.nodes.len() + i, mesh);
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); scene.nodes.len()];
    for (i, n) in scene.nodes.iter().enumerate() {
        if let Some(p) = n.parent {
            if p < children.len() {
                children[p].push(i);
            }
        }
    }
    let mut nodes = Vec::new();
    for (i, n) in scene.nodes.iter().enumerate() {
        let mut obj = Map::new();
        obj.insert("name".into(), json!(n.name));
        obj.insert("matrix".into(), json!(n.local.to_cols_array()));
        if !children[i].is_empty() {
            obj.insert("children".into(), json!(children[i]));
        }
        if let Some(m) = mesh_of_node.get(&i) {
            obj.insert("mesh".into(), json!(m));
        }
        if let Some(s) = n.skin {
            obj.insert("skin".into(), json!(s));
        }
        nodes.push(Value::Object(obj));
    }
    for (i, pi) in loose.iter().enumerate() {
        let n = scene.nodes.len() + i;
        let mut obj = Map::new();
        obj.insert("name".into(), json!(scene.primitives[*pi].name));
        if let Some(m) = mesh_of_node.get(&n) {
            obj.insert("mesh".into(), json!(m));
        }
        nodes.push(Value::Object(obj));
    }

    let roots: Vec<usize> = (0..scene.nodes.len())
        .filter(|i| scene.nodes[*i].parent.is_none())
        .collect();
    let mut preset = Map::new();
    for (name, binds) in expressions {
        let binds: Vec<Value> = binds
            .iter()
            .filter(|(node, _)| *node < scene.nodes.len())
            .map(|(node, index)| json!({ "node": node, "index": index, "weight": 1.0 }))
            .collect();
        preset.insert(name.clone(), json!({ "morphTargetBinds": binds }));
    }
    let meta = &scene.meta;
    let vrm = json!({
        "specVersion": "1.0",
        "meta": {
            "name": meta.name.as_deref().unwrap_or(&scene.name),
            "version": meta.version.as_deref().unwrap_or("1.0.0"),
            "authors": meta.authors,
            "others": {},
        },
        "humanoid": { "humanBones": human_bones(scene) },
        "firstPerson": {},
        "lookAt": { "type": "bone", "offsetFromHeadBone": [0.0, 0.0, 0.0] },
        "expressions": { "preset": Value::Object(preset), "custom": {} },
    });

    let root = json!({
        "asset": { "version": "2.0", "generator": "vroid-rs", "copyright": "VRoid Studio avatar (edited)" },
        "scene": 0,
        "scenes": [{ "nodes": roots }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": [{ "magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497 }],
        "skins": skins,
        "accessors": w.accessors,
        "bufferViews": w.views,
        "buffers": [{ "byteLength": w.bytes.len() }],
        "extensionsUsed": extensions_used,
        "extensions": { "VRMC_vrm": vrm },
    });

    let mut json_bytes = serde_json::to_vec(&root)?;
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let mut bin = w.bytes;
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6c67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004e_4942u32.to_le_bytes());
    out.extend_from_slice(&bin);

    let path = path.as_ref();
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    std::fs::write(path, &out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn build_mesh(w: &mut Writer, scene: &Scene, prims: &[usize]) -> Value {
    let mut json_prims = Vec::new();
    let target_count = prims
        .iter()
        .map(|&i| scene.primitives[i].morph_positions.len())
        .max()
        .unwrap_or(0);
    for &pi in prims {
        let p = &scene.primitives[pi];
        let mut pos = Vec::with_capacity(p.vertices.len());
        let mut nrm = Vec::with_capacity(p.vertices.len());
        let mut uv = Vec::with_capacity(p.vertices.len());
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in &p.vertices {
            pos.push(v.pos);
            nrm.push(v.normal);
            uv.push(v.uv);
            for k in 0..3 {
                lo[k] = lo[k].min(v.pos[k]);
                hi[k] = hi[k].max(v.pos[k]);
            }
        }
        let view = w.view(bytemuck::cast_slice(&pos), Some(ARRAY_BUFFER));
        let pos_acc = w.accessor(
            view,
            5126,
            "VEC3",
            pos.len(),
            Some(lo.to_vec()),
            Some(hi.to_vec()),
        );
        let view = w.view(bytemuck::cast_slice(&nrm), Some(ARRAY_BUFFER));
        let nrm_acc = w.accessor(view, 5126, "VEC3", nrm.len(), None, None);
        let view = w.view(bytemuck::cast_slice(&uv), Some(ARRAY_BUFFER));
        let uv_acc = w.accessor(view, 5126, "VEC2", uv.len(), None, None);

        let mut attrs = Map::new();
        attrs.insert("POSITION".into(), json!(pos_acc));
        attrs.insert("NORMAL".into(), json!(nrm_acc));
        attrs.insert("TEXCOORD_0".into(), json!(uv_acc));
        if let (Some(skin), Some(si)) = (&p.skin, p.skin_index) {
            if scene.skins.get(si).is_some() {
                let joints: Vec<[u16; 4]> = skin.joints.clone();
                let view = w.view(bytemuck::cast_slice(&joints), Some(ARRAY_BUFFER));
                let a = w.accessor(view, 5123, "VEC4", joints.len(), None, None);
                attrs.insert("JOINTS_0".into(), json!(a));
                let view = w.view(bytemuck::cast_slice(&skin.weights), Some(ARRAY_BUFFER));
                let a = w.accessor(view, 5126, "VEC4", skin.weights.len(), None, None);
                attrs.insert("WEIGHTS_0".into(), json!(a));
            }
        }

        let (indices, component) = if p.vertices.len() <= u16::MAX as usize {
            let idx: Vec<u16> = p.indices.iter().map(|i| *i as u16).collect();
            (bytemuck::cast_slice(&idx).to_vec(), 5123)
        } else {
            let idx: Vec<u32> = p.indices.clone();
            (bytemuck::cast_slice(&idx).to_vec(), 5125)
        };
        let view = w.view(&indices, Some(ELEMENT_ARRAY_BUFFER));
        let idx_acc = w.accessor(view, component, "SCALAR", p.indices.len(), None, None);

        let mut jp = Map::new();
        jp.insert("attributes".into(), Value::Object(attrs));
        jp.insert("indices".into(), json!(idx_acc));
        jp.insert("material".into(), json!(p.material));
        if target_count > 0 {
            let mut targets = Vec::new();
            for t in 0..target_count {
                let deltas: Vec<[f32; 3]> = p
                    .morph_positions
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| vec![[0.0; 3]; p.vertices.len()]);
                let view = w.view(bytemuck::cast_slice(&deltas), Some(ARRAY_BUFFER));
                let a = w.accessor(view, 5126, "VEC3", deltas.len(), None, None);
                targets.push(json!({ "POSITION": a }));
            }
            jp.insert("targets".into(), json!(targets));
        }
        json_prims.push(Value::Object(jp));
    }
    let names: Vec<String> = prims
        .first()
        .map(|&i| scene.primitives[i].morph_names.clone())
        .unwrap_or_default();
    let mut mesh = Map::new();
    mesh.insert("name".into(), json!(scene.primitives[prims[0]].name));
    mesh.insert("primitives".into(), json!(json_prims));
    if target_count > 0 {
        mesh.insert("extras".into(), json!({ "targetNames": names }));
    }
    Value::Object(mesh)
}

/// Build VRM expression binds from the morph names actually present in the
/// scene. Each preset maps to the morph deltas whose name ends with the
/// VRoid expression tag.
pub fn expression_binds(
    scene: &Scene,
    presets: &[(&str, &[&str])],
) -> BTreeMap<String, Vec<(usize, usize)>> {
    let mut out = BTreeMap::new();
    for (name, tags) in presets {
        let mut binds = Vec::new();
        for p in &scene.primitives {
            let Some(node) = p.node else { continue };
            for (i, morph) in p.morph_names.iter().enumerate() {
                if tags.iter().any(|t| morph.ends_with(t)) {
                    binds.push((node, i));
                }
            }
        }
        binds.sort_unstable();
        binds.dedup();
        out.insert((*name).to_string(), binds);
    }
    out
}
