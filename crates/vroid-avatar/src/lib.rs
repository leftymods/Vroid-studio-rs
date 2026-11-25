//! vroid-avatar — the editing document for a VRoid-style avatar.
//!
//! An `AvatarDoc` is a small, serialisable description of *changes* on top of a
//! base model: expression weights, raw morph weights and material overrides.
//! `apply` bakes those changes into a `Scene` so the renderer (or an exporter)
//! sees a plain mesh.

use anyhow::{Context, Result};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use vroid_model::Scene;

pub mod hair;

/// VRM 1.0 expression presets mapped onto VRoid's `Fcl_*` face morphs.
pub const EXPRESSIONS: &[(&str, &[&str])] = &[
    ("happy", &["Fcl_ALL_Joy"]),
    ("angry", &["Fcl_ALL_Angry"]),
    ("sad", &["Fcl_ALL_Sorrow"]),
    ("relaxed", &["Fcl_ALL_Fun"]),
    ("surprised", &["Fcl_ALL_Surprised"]),
    ("aa", &["Fcl_MTH_A"]),
    ("ih", &["Fcl_MTH_I"]),
    ("ou", &["Fcl_MTH_U"]),
    ("ee", &["Fcl_MTH_E"]),
    ("oh", &["Fcl_MTH_O"]),
    ("blink", &["Fcl_EYE_Close"]),
    ("blinkLeft", &["Fcl_EYE_Close_L"]),
    ("blinkRight", &["Fcl_EYE_Close_R"]),
    ("lookAt", &[]),
];

/// Convenience list of every preset name, in a stable order.
pub fn expression_names() -> impl Iterator<Item = &'static str> {
    EXPRESSIONS.iter().map(|(n, _)| *n)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MaterialOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_color: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shade_color: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outline_width: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outline_color: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AvatarDoc {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: String,
    /// Preset expression name -> weight in `[0, 1]`.
    #[serde(default)]
    pub expressions: BTreeMap<String, f32>,
    /// Raw morph target name (full or suffix) -> weight.
    #[serde(default)]
    pub morphs: BTreeMap<String, f32>,
    /// Material name -> override.
    #[serde(default)]
    pub materials: BTreeMap<String, MaterialOverride>,
}

impl AvatarDoc {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading avatar doc {}", path.display()))?;
        let doc: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing avatar doc {}", path.display()))?;
        Ok(doc)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).with_context(|| format!("writing avatar doc {}", path.display()))
    }

    /// Expand presets + raw morphs into a single morph-weight map. Preset
    /// weights are summed into their target morphs; explicit morph weights are
    /// added on top.
    pub fn resolve_morphs(&self) -> BTreeMap<String, f32> {
        let mut out: BTreeMap<String, f32> = BTreeMap::new();
        for (preset, &w) in &self.expressions {
            if w == 0.0 {
                continue;
            }
            if let Some((_, targets)) = EXPRESSIONS.iter().find(|(n, _)| n == preset) {
                for t in *targets {
                    *out.entry((*t).to_string()).or_insert(0.0) += w;
                }
            }
        }
        for (name, &w) in &self.morphs {
            *out.entry(name.clone()).or_insert(0.0) += w;
        }
        for w in out.values_mut() {
            *w = w.clamp(0.0, 1.0);
        }
        out
    }

    /// Bake morphs and material overrides into `scene`.
    pub fn apply(&self, scene: &mut Scene) {
        let weights = self.resolve_morphs();
        apply_morphs(scene, &weights);
        self.apply_materials(scene);
    }

    pub fn apply_materials(&self, scene: &mut Scene) {
        for mat in &mut scene.materials {
            let Some(o) = self.materials.get(&mat.name) else { continue };
            if let Some(c) = o.base_color {
                mat.base_color = c;
            }
            if let Some(c) = o.shade_color {
                mat.mtoon.shade_color = c;
            }
            if let Some(w) = o.outline_width {
                mat.mtoon.outline_width = w;
            }
            if let Some(c) = o.outline_color {
                mat.mtoon.outline_color = c;
            }
        }
    }
}

/// Match a morph name by exact value or by suffix (so `ALL_Joy` finds
/// `M_F00_000_00_Fcl_ALL_Joy`).
pub fn morph_matches(full: &str, query: &str) -> bool {
    full == query || full.ends_with(query) || full.contains(query)
}

/// Add scaled morph deltas onto a scene's bind-pose vertices.
pub fn apply_morphs(scene: &mut Scene, weights: &BTreeMap<String, f32>) {
    for prim in &mut scene.primitives {
        if prim.morph_positions.is_empty() || weights.is_empty() {
            continue;
        }
        let n = prim.vertices.len();
        let mut delta = vec![Vec3::ZERO; n];
        let mut any = false;
        for (i, name) in prim.morph_names.iter().enumerate() {
            if name.is_empty() || i >= prim.morph_positions.len() {
                continue;
            }
            let mut w = 0.0;
            for (q, v) in weights {
                if morph_matches(name, q) {
                    w += *v;
                }
            }
            if w == 0.0 {
                continue;
            }
            any = true;
            for (d, src) in delta.iter_mut().zip(&prim.morph_positions[i]) {
                *d += Vec3::from(*src) * w;
            }
        }
        if !any {
            continue;
        }
        for (v, d) in prim.vertices.iter_mut().zip(delta) {
            v.pos = (Vec3::from(v.pos) + d).to_array();
        }
    }
}

/// Parse a `name=value` list (`a=1,b=0.5`). Used by the CLI.
pub fn parse_weights(s: &str) -> Result<Vec<(String, f32)>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (k, v) = part
            .split_once('=')
            .with_context(|| format!("expected name=value, got `{part}`"))?;
        out.push((k.trim().to_string(), v.trim().parse::<f32>()?));
    }
    Ok(out)
}
