//! Facial proportion parameters (VRoid Studio style).
//!
//! Unlike expressions, these are *shape* edits: we move the vertices of the
//! face mesh itself. The face is a stack of primitives (skin, eyes, brows,
//! mouth, …) sharing one vertex layout; a region is located by the material
//! names of a primitive and every layer is displaced by the same per-vertex
//! delta so the decals stay glued to the skin.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use vroid_model::{Scene, Vertex};

/// Face sliders, each in `[-1, 1]` (0 = the base model).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceParams {
    pub eye_size: f32,
    pub eye_width: f32,
    pub eye_height: f32,
    pub eye_spacing: f32,
    pub eye_angle: f32,
    pub eye_y: f32,
    pub brow_size: f32,
    pub brow_y: f32,
    pub brow_angle: f32,
    pub nose_size: f32,
    pub nose_y: f32,
    pub mouth_size: f32,
    pub mouth_width: f32,
    pub mouth_y: f32,
}

impl FaceParams {
    pub fn is_neutral(&self) -> bool {
        [
            self.eye_size,
            self.eye_width,
            self.eye_height,
            self.eye_spacing,
            self.eye_angle,
            self.eye_y,
            self.brow_size,
            self.brow_y,
            self.brow_angle,
            self.nose_size,
            self.nose_y,
            self.mouth_size,
            self.mouth_width,
            self.mouth_y,
        ]
        .iter()
        .all(|v| v.abs() < 1e-4)
    }
}

#[derive(Clone, Copy, Debug)]
struct Region {
    center: Vec3,
    radius: Vec3,
}

impl Region {
    fn from_vertices(verts: &[[f32; 3]], indices: &[usize]) -> Option<Self> {
        if indices.is_empty() {
            return None;
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for &i in indices {
            if let Some(p) = verts.get(i) {
                let p = Vec3::from(*p);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        Some(Self {
            center: (lo + hi) * 0.5,
            radius: (hi - lo) * 0.5 + Vec3::splat(1e-4),
        })
    }

    fn weight(self, p: Vec3) -> f32 {
        let q = (p - self.center) / self.radius;
        let d = q.length_squared();
        if d >= 1.0 {
            return 0.0;
        }
        let t = 1.0 - d;
        t * t * (3.0 - 2.0 * t)
    }
}

fn is_face(scene: &Scene, pi: usize) -> bool {
    let p = &scene.primitives[pi];
    if !p.morph_positions.is_empty() {
        return true;
    }
    let n = p.name.to_ascii_lowercase();
    ["face", "eye", "iris", "brow", "mouth", "lash", "highlight"]
        .iter()
        .any(|k| n.contains(k))
}

fn material_indices(scene: &Scene, keys: &[&str]) -> Vec<usize> {
    let mut out = Vec::new();
    for (pi, p) in scene.primitives.iter().enumerate() {
        if !is_face(scene, pi) {
            continue;
        }
        let n = scene.materials[p.material].name.to_ascii_lowercase();
        if keys.iter().any(|k| n.contains(k)) {
            out.extend(p.indices.iter().map(|i| *i as usize));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn split_sides(verts: &[[f32; 3]], indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in indices {
        if verts.get(i).map(|p| p[0] < 0.0).unwrap_or(false) {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

fn scale(delta: &mut [Vec3], verts: &[Vertex], r: Region, f: Vec3) {
    for (d, v) in delta.iter_mut().zip(verts) {
        let p = Vec3::from(v.pos);
        *d += (p - r.center) * (f - Vec3::ONE) * r.weight(p);
    }
}

fn translate(delta: &mut [Vec3], verts: &[Vertex], r: Region, off: Vec3) {
    for (d, v) in delta.iter_mut().zip(verts) {
        let p = Vec3::from(v.pos);
        *d += off * r.weight(p);
    }
}

fn rotate(delta: &mut [Vec3], verts: &[Vertex], r: Region, angle: f32) {
    if angle.abs() < 1e-6 {
        return;
    }
    let (c, s) = (angle.cos(), angle.sin());
    for (d, v) in delta.iter_mut().zip(verts) {
        let p = Vec3::from(v.pos);
        let q = p - r.center;
        let rotated = Vec3::new(q.x * c - q.y * s, q.x * s + q.y * c, q.z);
        *d += (rotated - q) * r.weight(p);
    }
}

/// Apply face proportions to `scene` in place. Call once on a fresh clone.
pub fn apply(scene: &mut Scene, p: &FaceParams) {
    if p.is_neutral() {
        return;
    }
    let face: Vec<usize> = (0..scene.primitives.len())
        .filter(|i| is_face(scene, *i))
        .collect();
    if face.is_empty() {
        return;
    }
    let verts = scene.primitives[face[0]].vertices.clone();
    let face_pos: Vec<[f32; 3]> = verts.iter().map(|v| v.pos).collect();
    let mut delta = vec![Vec3::ZERO; verts.len()];

    let eye_keys = ["eye", "iris", "lash", "highlight"];
    let eye_idx = material_indices(scene, &eye_keys);
    let (eye_l, eye_r) = split_sides(&face_pos, &eye_idx);
    for (idx, sign) in [(eye_l, -1.0), (eye_r, 1.0)] {
        let Some(r) = Region::from_vertices(&face_pos, &idx) else {
            continue;
        };
        let size = 1.0 + 0.34 * p.eye_size;
        scale(
            &mut delta,
            &verts,
            r,
            Vec3::new(size + 0.26 * p.eye_width, size + 0.26 * p.eye_height, size),
        );
        translate(
            &mut delta,
            &verts,
            r,
            Vec3::new(sign * 0.012 * p.eye_spacing, 0.010 * p.eye_y, 0.0),
        );
        rotate(&mut delta, &verts, r, sign * 0.22 * p.eye_angle);
    }

    let brow_idx = material_indices(scene, &["brow"]);
    let (brow_l, brow_r) = split_sides(&face_pos, &brow_idx);
    for (idx, sign) in [(brow_l, -1.0), (brow_r, 1.0)] {
        let Some(r) = Region::from_vertices(&face_pos, &idx) else {
            continue;
        };
        let s = 1.0 + 0.45 * p.brow_size;
        scale(&mut delta, &verts, r, Vec3::new(s, s, 1.0));
        translate(&mut delta, &verts, r, Vec3::new(0.0, 0.010 * p.brow_y, 0.0));
        rotate(&mut delta, &verts, r, sign * 0.26 * p.brow_angle);
    }

    let mouth_idx = material_indices(scene, &["mouth"]);
    if let Some(r) = Region::from_vertices(&face_pos, &mouth_idx) {
        let s = 1.0 + 0.38 * p.mouth_size;
        scale(
            &mut delta,
            &verts,
            r,
            Vec3::new(s + 0.30 * p.mouth_width, s, s),
        );
        translate(
            &mut delta,
            &verts,
            r,
            Vec3::new(0.0, 0.012 * p.mouth_y, 0.0),
        );
    }

    let nose_idx = material_indices(scene, &["skin"]);
    let nose_idx: Vec<usize> = nose_idx
        .into_iter()
        .filter(|&i| {
            face_pos
                .get(i)
                .map(|v| v[0].abs() < 0.026 && v[2] > 0.045)
                .unwrap_or(false)
        })
        .collect();
    if let Some(r) = Region::from_vertices(&face_pos, &nose_idx) {
        let s = 1.0 + 0.40 * p.nose_size;
        scale(&mut delta, &verts, r, Vec3::splat(s));
        translate(&mut delta, &verts, r, Vec3::new(0.0, 0.012 * p.nose_y, 0.0));
    }

    for pi in face {
        let prim = &mut scene.primitives[pi];
        for (v, d) in prim.vertices.iter_mut().zip(&delta) {
            v.pos = (Vec3::from(v.pos) + *d).to_array();
        }
    }
    scene.recompute_bounds();
}
