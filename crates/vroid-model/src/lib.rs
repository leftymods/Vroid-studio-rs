//! vroid-model — glTF 2.0 / VRM loader producing a render-ready scene.
//!
//! Handles: node hierarchy with world transforms, mesh primitives, materials
//! (PBR + MToon), textures, and VRM metadata. Skinning/morph data is parsed and
//! exposed for later animation work, but the v0 renderer draws the bind pose.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use glam::{Mat3, Mat4, Vec3};

pub mod export;
pub mod mtoon;

pub use mtoon::MToon;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub const STRIDE: u64 = 32;
}

#[derive(Clone, Debug)]
pub struct Texture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub base_color_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub mtoon: MToon,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0; 4],
            base_color_tex: None,
            normal_tex: None,
            emissive: [0.0; 3],
            metallic: 0.0,
            roughness: 0.8,
            double_sided: false,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            mtoon: MToon::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Primitive {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material: usize,
    /// Bind-pose skinning data (JOINTS_0 / WEIGHTS_0), if present.
    pub skin: Option<SkinWeights>,
    /// Index into `Scene::skins` when this primitive is skinned.
    pub skin_index: Option<usize>,
    /// Source glTF node this primitive came from (`None` for generated meshes).
    pub node: Option<usize>,
    /// Morph targets as per-vertex position deltas.
    pub morph_positions: Vec<Vec<[f32; 3]>>,
    /// Morph target names (from mesh extras `targetNames`), parallel to
    /// `morph_positions`.
    pub morph_names: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SkinWeights {
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}

#[derive(Clone, Debug)]
pub struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
}

#[derive(Clone, Debug, Default)]
pub struct VrmMeta {
    pub name: Option<String>,
    pub version: Option<String>,
    pub authors: Vec<String>,
    pub vrm_version: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub local: Mat4,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub name: String,
    pub generator: String,
    pub primitives: Vec<Primitive>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub skins: Vec<Skin>,
    pub meta: VrmMeta,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    /// World transform per node index (for animation/posing later).
    pub node_world: Vec<Mat4>,
    /// Node hierarchy (name, parent, local bind transform).
    pub nodes: Vec<Node>,
}

impl Scene {
    pub fn center(&self) -> Vec3 {
        (self.bounds_min + self.bounds_max) * 0.5
    }
    pub fn radius(&self) -> f32 {
        (self.bounds_max - self.bounds_min).length() * 0.5
    }
    pub fn height(&self) -> f32 {
        self.bounds_max.y - self.bounds_min.y
    }
    pub fn vertex_count(&self) -> usize {
        self.primitives.iter().map(|p| p.vertices.len()).sum()
    }
    pub fn triangle_count(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }
    /// Recompute world matrices from the node hierarchy.
    pub fn world_matrices(&self) -> Vec<Mat4> {
        let mut world = vec![Mat4::IDENTITY; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let mut chain = vec![i];
            let mut cur = self.nodes[i].parent;
            while let Some(p) = cur {
                chain.push(p);
                cur = self.nodes[p].parent;
            }
            let mut m = Mat4::IDENTITY;
            for &idx in chain.iter().rev() {
                m *= self.nodes[idx].local;
            }
            world[i] = m;
        }
        world
    }
    /// Recompute the axis-aligned bounds from the current vertex positions.
    pub fn recompute_bounds(&mut self) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &self.primitives {
            for v in &p.vertices {
                let p3 = Vec3::from(v.pos);
                lo = lo.min(p3);
                hi = hi.max(p3);
            }
        }
        if lo.is_finite() && hi.is_finite() {
            self.bounds_min = lo;
            self.bounds_max = hi;
        }
    }
}

/// Load a `.glb`/`.gltf`/`.vrm` file.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<Scene> {
    let path = path.as_ref();
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("importing {}", path.display()))?;
    let gltf_json = read_glb_json(path);

    let mut textures = Vec::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        textures.push(Texture {
            name: format!("image_{i}"),
            width: img.width,
            height: img.height,
            rgba: to_rgba8(img),
        });
    }

    let mut materials = Vec::with_capacity(doc.materials().len());
    for (i, m) in doc.materials().enumerate() {
        materials.push(load_material(&m, &gltf_json, i));
    }
    if materials.is_empty() {
        materials.push(Material::default());
    }

    let mut skins = Vec::new();
    for s in doc.skins() {
        let joints = s.joints().map(|j| j.index()).collect::<Vec<_>>();
        let inverse_bind = s
            .reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()))
            .read_inverse_bind_matrices()
            .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![Mat4::IDENTITY; joints.len()]);
        skins.push(Skin {
            name: s.name().unwrap_or("skin").to_string(),
            joints,
            inverse_bind,
        });
    }

    let node_count = doc.nodes().len();
    let mut parent_of = vec![None; node_count];
    for node in doc.nodes() {
        for child in node.children() {
            parent_of[child.index()] = Some(node.index());
        }
    }
    let mut nodes = Vec::with_capacity(node_count);
    for node in doc.nodes() {
        nodes.push(Node {
            name: node.name().unwrap_or("node").to_string(),
            parent: parent_of[node.index()],
            local: Mat4::from_cols_array_2d(&node.transform().matrix()),
            mesh: node.mesh().map(|m| m.index()),
            skin: node.skin().map(|s| s.index()),
        });
    }
    let mut node_world = vec![Mat4::IDENTITY; node_count];
    for node in doc.nodes() {
        let mut chain = vec![node.index()];
        let mut cur = parent_of[node.index()];
        while let Some(p) = cur {
            chain.push(p);
            cur = parent_of[p];
        }
        let mut m = Mat4::IDENTITY;
        for idx in chain.iter().rev() {
            let n = doc.nodes().nth(*idx).unwrap();
            m = m * Mat4::from_cols_array_2d(&n.transform().matrix());
        }
        node_world[node.index()] = m;
    }

    let mut primitives = Vec::new();
    let mut bounds_min = Vec3::splat(f32::INFINITY);
    let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);

    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let mesh_idx = mesh.index();
        let target_names: Vec<String> = gltf_json
            .as_ref()
            .and_then(|j| {
                j.get("meshes")?
                    .get(mesh_idx)?
                    .get("extras")?
                    .get("targetNames")?
                    .as_array()
                    .cloned()
            })
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let skin_index = node.skin().map(|s| s.index());
        // glTF: a skinned mesh is defined in skin space; its node transform is ignored.
        let world = if skin_index.is_some() {
            Mat4::IDENTITY
        } else {
            node_world[node.index()]
        };
        let normal_mat = Mat3::from_mat4(world).inverse().transpose();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            let skin = match (reader.read_joints(0), reader.read_weights(0)) {
                (Some(j), Some(w)) => Some(SkinWeights {
                    joints: j.into_u16().collect(),
                    weights: w.into_f32().collect(),
                }),
                _ => None,
            };

            let mut morph_positions = Vec::new();
            for (pos, _nrm, _tan) in reader.read_morph_targets() {
                if let Some(deltas) = pos {
                    morph_positions.push(deltas.collect());
                }
            }

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                let wp = world.transform_point3(Vec3::from(positions[i]));
                let wn = normal_mat * Vec3::from(normals[i]);
                let wn = if wn.length_squared() > 1e-12 {
                    wn.normalize()
                } else {
                    Vec3::Y
                };
                bounds_min = bounds_min.min(wp);
                bounds_max = bounds_max.max(wp);
                vertices.push(Vertex {
                    pos: wp.to_array(),
                    normal: wn.to_array(),
                    uv: uvs[i],
                });
            }

            let material = prim
                .material()
                .index()
                .unwrap_or(0)
                .min(materials.len() - 1);
            let mut morph_names = target_names.clone();
            morph_names.resize(morph_positions.len(), String::new());
            primitives.push(Primitive {
                name: format!("{}#{:?}", mesh.name().unwrap_or("mesh"), prim.mode()),
                vertices,
                indices,
                material,
                skin,
                skin_index,
                node: Some(node.index()),
                morph_positions,
                morph_names,
            });
        }
    }

    if !bounds_min.is_finite() || !bounds_max.is_finite() {
        bounds_min = Vec3::splat(-1.0);
        bounds_max = Vec3::splat(1.0);
    }

    let meta = load_meta(&gltf_json);
    let name = doc
        .default_scene()
        .and_then(|s| s.name().map(str::to_string))
        .unwrap_or_else(|| "scene".to_string());

    Ok(Scene {
        name,
        generator: doc.as_json().asset.generator.clone().unwrap_or_default(),
        primitives,
        materials,
        textures,
        skins,
        meta,
        bounds_min,
        bounds_max,
        node_world,
        nodes,
    })
}

fn load_material(m: &gltf::Material, json: &Option<serde_json::Value>, index: usize) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let mut mat = Material {
        name: m.name().unwrap_or("material").to_string(),
        base_color: pbr.base_color_factor(),
        base_color_tex: pbr
            .base_color_texture()
            .map(|t| t.texture().source().index()),
        normal_tex: m.normal_texture().map(|t| t.texture().source().index()),
        emissive: m.emissive_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        double_sided: m.double_sided(),
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        mtoon: MToon::default(),
    };
    if let Some(v) = json {
        mat.mtoon = mtoon::from_gltf_json(v, index);
    }
    mat
}

fn load_meta(json: &Option<serde_json::Value>) -> VrmMeta {
    let Some(v) = json else {
        return VrmMeta::default();
    };
    let ext = &v["extensions"];
    if let Some(vrm1) = ext.get("VRMC_vrm") {
        let meta = &vrm1["meta"];
        return VrmMeta {
            name: meta["name"].as_str().map(str::to_string),
            version: meta["version"].as_str().map(str::to_string),
            authors: meta["authors"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            vrm_version: Some("1.0".into()),
        };
    }
    if let Some(vrm0) = ext.get("VRM") {
        let meta = &vrm0["meta"];
        return VrmMeta {
            name: meta["title"].as_str().map(str::to_string),
            version: meta["version"].as_str().map(str::to_string),
            authors: meta["author"]
                .as_str()
                .map(|s| vec![s.to_string()])
                .unwrap_or_default(),
            vrm_version: Some("0.x".into()),
        };
    }
    VrmMeta::default()
}

/// Parse the JSON chunk of a GLB (or the whole .gltf file) for extensions.
fn read_glb_json(path: &std::path::Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() >= 20 && &bytes[0..4] == b"glTF" {
        let chunk_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let chunk_type = &bytes[16..20];
        if chunk_type == b"JSON" && 20 + chunk_len <= bytes.len() {
            return serde_json::from_slice(&bytes[20..20 + chunk_len]).ok();
        }
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

fn to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format as F;
    let n = (img.width * img.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    match img.format {
        F::R8 => {
            for &r in &img.pixels {
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        F::R8G8 => {
            for c in img.pixels.chunks_exact(2) {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
        }
        F::R8G8B8 => {
            for c in img.pixels.chunks_exact(3) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        F::R8G8B8A8 => out.extend_from_slice(&img.pixels),
        F::R16 => {
            for c in img.pixels.chunks_exact(2) {
                let v = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        F::R16G16B16 => {
            for c in img.pixels.chunks_exact(6) {
                out.extend_from_slice(&[c[1], c[3], c[5], 255]);
            }
        }
        F::R16G16B16A16 => {
            for c in img.pixels.chunks_exact(8) {
                out.extend_from_slice(&[c[1], c[3], c[5], c[7]]);
            }
        }
        _ => out.extend_from_slice(&vec![255u8; n * 4]),
    }
    out
}

/// Convenience: group textures referenced by materials (unused for now).
pub fn texture_usage(scene: &Scene) -> BTreeMap<usize, Vec<String>> {
    let mut map: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for m in &scene.materials {
        if let Some(t) = m.base_color_tex {
            map.entry(t).or_default().push(format!("{}.base", m.name));
        }
        if let Some(t) = m.normal_tex {
            map.entry(t).or_default().push(format!("{}.normal", m.name));
        }
    }
    map
}
