//! Body proportions driven by bone transforms.
//!
//! VRoid's body parameters are not blendshapes: the body mesh is skinned to a
//! humanoid skeleton and each slider edits bone transforms. We reproduce that by
//! editing the bind-local matrices of the relevant bones, recomputing the world
//! hierarchy and then re-skinning every skinned primitive on the CPU.

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use vroid_model::Scene;

/// Normalised body sliders in `[-1, 1]` (0 = the base model).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BodyParams {
    #[serde(default)]
    pub height: f32,
    #[serde(default)]
    pub head_size: f32,
    #[serde(default)]
    pub neck: f32,
    #[serde(default)]
    pub arms: f32,
    #[serde(default)]
    pub legs: f32,
    #[serde(default)]
    pub shoulders: f32,
    #[serde(default)]
    pub hips: f32,
    #[serde(default)]
    pub waist: f32,
    #[serde(default)]
    pub bust: f32,
    #[serde(default)]
    pub thickness: f32,
}

impl BodyParams {
    pub fn is_neutral(&self) -> bool {
        [
            self.height,
            self.head_size,
            self.neck,
            self.arms,
            self.legs,
            self.shoulders,
            self.hips,
            self.waist,
            self.bust,
            self.thickness,
        ]
        .iter()
        .all(|v| v.abs() < 1e-4)
    }
}

fn find(scene: &Scene, name: &str) -> Option<usize> {
    scene.nodes.iter().position(|n| n.name == name)
}

fn translate(local: &mut Mat4, d: Vec3) {
    local.w_axis = (local.w_axis.truncate() + d).extend(1.0);
}

fn scale_translation(local: &mut Mat4, s: Vec3) {
    local.w_axis = (local.w_axis.truncate() * s).extend(1.0);
}

fn scale_basis(local: &mut Mat4, s: Vec3) {
    local.x_axis = (local.x_axis.truncate() * s.x).extend(0.0);
    local.y_axis = (local.y_axis.truncate() * s.y).extend(0.0);
    local.z_axis = (local.z_axis.truncate() * s.z).extend(0.0);
}

fn worlds(scene: &Scene, locals: &[Mat4]) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; locals.len()];
    for i in 0..locals.len() {
        let mut chain = vec![i];
        let mut cur = scene.nodes[i].parent;
        while let Some(p) = cur {
            chain.push(p);
            cur = scene.nodes[p].parent;
        }
        let mut m = Mat4::IDENTITY;
        for &idx in chain.iter().rev() {
            m *= locals[idx];
        }
        world[i] = m;
    }
    world
}

/// Hip-to-foot distance in the bind pose, used to keep feet planted.
const LEG_LEN: f32 = 0.725;

/// Apply the body parameters to `scene` in place (re-skins all skinned
/// primitives). Call on a freshly cloned scene, not repeatedly on the same one.
pub fn apply(scene: &mut Scene, p: &BodyParams) {
    if p.is_neutral() || scene.skins.is_empty() {
        return;
    }

    let mut locals: Vec<Mat4> = scene.nodes.iter().map(|n| n.local).collect();

    let leg_k = 1.0 + 0.20 * p.legs + 0.16 * p.height;
    let lift = (leg_k - 1.0) * LEG_LEN;

    if let Some(i) = find(scene, "J_Bip_C_Hips") {
        translate(&mut locals[i], Vec3::new(0.0, lift, 0.0));
    }
    for name in ["J_Bip_L_LowerLeg", "J_Bip_R_LowerLeg", "J_Bip_L_Foot", "J_Bip_R_Foot"] {
        if let Some(i) = find(scene, name) {
            scale_translation(&mut locals[i], Vec3::splat(leg_k));
        }
    }

    let arm_k = 1.0 + 0.22 * p.arms;
    for name in ["J_Bip_L_LowerArm", "J_Bip_R_LowerArm", "J_Bip_L_Hand", "J_Bip_R_Hand"] {
        if let Some(i) = find(scene, name) {
            scale_translation(&mut locals[i], Vec3::splat(arm_k));
        }
    }

    let neck_k = 1.0 + 0.30 * p.neck;
    if let Some(i) = find(scene, "J_Bip_C_Neck") {
        scale_translation(&mut locals[i], Vec3::splat(neck_k));
    }

    let head_k = 1.0 + 0.18 * p.head_size;
    if let Some(i) = find(scene, "J_Bip_C_Head") {
        scale_basis(&mut locals[i], Vec3::splat(head_k));
    }

    let shoulder_k = 1.0 + 0.35 * p.shoulders;
    for name in ["J_Bip_L_Shoulder", "J_Bip_R_Shoulder"] {
        if let Some(i) = find(scene, name) {
            scale_translation(&mut locals[i], Vec3::new(shoulder_k, 1.0, 1.0));
        }
    }

    let hip_k = 1.0 + 0.30 * p.hips;
    for name in ["J_Bip_L_UpperLeg", "J_Bip_R_UpperLeg"] {
        if let Some(i) = find(scene, name) {
            scale_translation(&mut locals[i], Vec3::new(hip_k, 1.0, 1.0));
        }
    }

    let waist_k = 1.0 + 0.20 * p.waist;
    for name in ["J_Bip_C_Spine", "J_Bip_C_Chest"] {
        if let Some(i) = find(scene, name) {
            scale_basis(&mut locals[i], Vec3::new(waist_k, 1.0, waist_k));
        }
    }

    let thick_k = 1.0 + 0.12 * p.thickness;
    for name in ["J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest"] {
        if let Some(i) = find(scene, name) {
            scale_basis(&mut locals[i], Vec3::new(thick_k, 1.0, thick_k));
        }
    }

    let bust_k = 1.0 + 0.40 * p.bust;
    for name in ["J_Sec_L_Bust1", "J_Sec_R_Bust1", "J_Sec_L_Bust2", "J_Sec_R_Bust2"] {
        if let Some(i) = find(scene, name) {
            scale_basis(&mut locals[i], Vec3::splat(bust_k));
        }
    }

    let world = worlds(scene, &locals);
    scene.node_world = world.clone();

    for prim in &mut scene.primitives {
        let (Some(si), Some(sw)) = (prim.skin_index, &prim.skin) else { continue };
        let Some(skin) = scene.skins.get(si) else { continue };
        let joint_mats: Vec<Mat4> = skin
            .joints
            .iter()
            .enumerate()
            .map(|(k, &node)| {
                world[node] * skin.inverse_bind.get(k).copied().unwrap_or(Mat4::IDENTITY)
            })
            .collect();
        for (i, v) in prim.vertices.iter_mut().enumerate() {
            let j = sw.joints.get(i).copied().unwrap_or([0; 4]);
            let w = sw.weights.get(i).copied().unwrap_or([0.0; 4]);
            let mut pos = Vec3::ZERO;
            let mut nrm = Vec3::ZERO;
            let mut total = 0.0;
            for c in 0..4 {
                let wc = w[c];
                if wc <= 0.0 {
                    continue;
                }
                let Some(m) = joint_mats.get(j[c] as usize).copied() else { continue };
                pos += m.transform_point3(Vec3::from(v.pos)) * wc;
                nrm += m.transform_vector3(Vec3::from(v.normal)) * wc;
                total += wc;
            }
            if total > 1e-6 {
                v.pos = (pos / total).to_array();
                v.normal = nrm.normalize_or_zero().to_array();
            }
        }
    }

    scene.recompute_bounds();
}
