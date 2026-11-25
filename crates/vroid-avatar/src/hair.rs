//! Procedural anime hairstyle for the VRoid base head.
//!
//! The head faces -Z.  The scalp is an ellipsoid around `C` with the radii
//! below (measured from the extracted base avatar).  The style is a
//! chin-length bob: a scalp cap, a layered fringe, face-framing side locks and
//! a full back sheet.  Cards are flat and double sided; the shader flips the
//! normal on back faces, so one card per strand is enough.

use glam::Vec3;
use vroid_model::{AlphaMode, Primitive, Scene, Vertex};

const C: Vec3 = Vec3::new(0.0, 1.395, -0.030);
const RX: f32 = 0.099;
const RY: f32 = 0.133;
const RZ: f32 = 0.101;
const TAU: f32 = std::f32::consts::TAU;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Hairline: high over the face, low around the sides and the nape.
fn theta_max(phi: f32) -> f32 {
    let d = (phi - std::f32::consts::PI).abs();
    1.0 + 1.05 * smoothstep(0.5, 1.55, d)
}

#[derive(Clone, Copy)]
enum Axis {
    NegZ,
    PosZ,
    PosX,
    NegX,
}

struct Builder {
    verts: Vec<Vertex>,
    idx: Vec<u32>,
}

impl Builder {
    fn new() -> Self {
        Self { verts: Vec::new(), idx: Vec::new() }
    }

    fn add(&mut self, p: Vec3, uv: [f32; 2]) -> u32 {
        self.verts.push(Vertex { pos: p.to_array(), normal: [0.0; 3], uv });
        (self.verts.len() - 1) as u32
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.idx.extend_from_slice(&[a, b, c, a, c, d]);
    }

    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.idx.extend_from_slice(&[a, b, c]);
    }
}

fn add_cap(b: &mut Builder) {
    const NP: usize = 36;
    const NT: usize = 14;
    let pole = b.add(Vec3::new(C.x, C.y + RY, C.z), [0.5, 0.0]);
    let mut grid: Vec<Vec<u32>> = Vec::with_capacity(NP);
    for j in 0..NP {
        let phi = TAU * j as f32 / NP as f32;
        let thm = theta_max(phi);
        let mut col = Vec::with_capacity(NT);
        for i in 1..=NT {
            let th = thm * i as f32 / NT as f32;
            let mut p = Vec3::new(
                C.x + RX * th.sin() * phi.sin(),
                C.y + RY * th.cos(),
                C.z + RZ * th.sin() * phi.cos(),
            );
            let k = 1.0 + 0.05 * (1.0 - i as f32 / NT as f32 * 2.0).max(0.0);
            p = C + (p - C) * k;
            col.push(b.add(p, [j as f32 / NP as f32, i as f32 / NT as f32]));
        }
        grid.push(col);
    }
    for j in 0..NP {
        let jn = (j + 1) % NP;
        b.tri(pole, grid[j][0], grid[jn][0]);
        for i in 0..NT - 1 {
            b.quad(grid[j][i], grid[j][i + 1], grid[jn][i + 1], grid[jn][i]);
        }
    }
}

fn add_card(b: &mut Builder, spine: &[Vec3], widths: &[f32], axis: Axis, bulge: f32) {
    let n = spine.len();
    let mut l = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for k in 0..n {
        let p = spine[k];
        let w = widths[k];
        let u = k as f32 / (n - 1) as f32;
        let (pl, pm, pr) = match axis {
            Axis::NegZ => (
                p + Vec3::new(-w, 0.0, 0.0),
                p + Vec3::new(0.0, 0.0, -bulge),
                p + Vec3::new(w, 0.0, 0.0),
            ),
            Axis::PosZ => (
                p + Vec3::new(-w, 0.0, 0.0),
                p + Vec3::new(0.0, 0.0, bulge),
                p + Vec3::new(w, 0.0, 0.0),
            ),
            Axis::PosX => (
                p + Vec3::new(0.0, 0.0, -w),
                p + Vec3::new(bulge, 0.0, 0.0),
                p + Vec3::new(0.0, 0.0, w),
            ),
            Axis::NegX => (
                p + Vec3::new(0.0, 0.0, -w),
                p + Vec3::new(-bulge, 0.0, 0.0),
                p + Vec3::new(0.0, 0.0, w),
            ),
        };
        l.push(b.add(pl, [0.0, u]));
        m.push(b.add(pm, [0.5, u]));
        r.push(b.add(pr, [1.0, u]));
    }
    for k in 0..n - 1 {
        b.quad(l[k], l[k + 1], m[k + 1], m[k]);
        b.quad(m[k], m[k + 1], r[k + 1], r[k]);
    }
}

fn spine_curve(p0: Vec3, p1: Vec3, bow: f32, n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let s = 1.0 - t;
            Vec3::new(
                p0.x * s + p1.x * t + bow * (std::f32::consts::PI * t).sin(),
                p0.y * s + p1.y * t,
                p0.z * s + p1.z * t,
            )
        })
        .collect()
}

fn add_bangs(b: &mut Builder) {
    // Nine overlapping locks.  The first row sits inside the scalp cap
    // (z behind the cap surface) so the top edge is never visible; the locks
    // then curve out over the forehead and hang to the brow line.
    let n = 9usize;
    let mut xs = Vec::with_capacity(n);
    let mut bot = Vec::with_capacity(n);
    let mut zoff = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32; // 0..1
        xs.push(-0.082 + 0.164 * t);
        // alternate lengths, longest near the temples
        let wave = (i as f32 * 1.7).sin() * 0.008;
        let edge = (t - 0.5).abs() * 2.0; // 0 centre, 1 temples
        bot.push(1.436 - 0.010 * edge + wave);
        zoff.push(((i as f32 * 2.3).cos()) * 0.0015);
    }
    for i in 0..n {
        let x = xs[i];
        let y1 = bot[i];
        let zo = zoff[i];
        let top = 1.500;
        let zt = -0.048 + zo;      // inside the cap (hidden edge)
        let z1 = -0.103 + zo;      // clearly in front of the brow
        let mut spine = Vec::with_capacity(5);
        let mut widths = Vec::with_capacity(5);
        let ws = [0.023, 0.025, 0.023, 0.015, 0.004];
        for k in 0..5 {
            let t = k as f32 / 4.0;
            let z = zt + (z1 - zt) * t.powf(0.35);
            let y = top + (y1 - top) * t;
            spine.push(Vec3::new(x + 0.004 * t * t, y, z));
            widths.push(ws[k]);
        }
        add_card(b, &spine, &widths, Axis::NegZ, 0.004);
    }
}

fn add_center_part(b: &mut Builder) {
    let spine = [
        Vec3::new(0.0, 1.505, -0.020),
        Vec3::new(0.010, 1.490, -0.060),
        Vec3::new(0.026, 1.474, -0.092),
    ];
    add_card(b, &spine, &[0.020, 0.018, 0.010], Axis::NegZ, 0.004);
}

fn add_side_locks(b: &mut Builder) {
    for s in [-1.0f32, 1.0] {
        let p0 = Vec3::new(s * 0.094, 1.476, -0.046);
        let p1 = Vec3::new(s * 0.120, 1.232, -0.006);
        let sp = spine_curve(p0, p1, s * 0.006, 7);
        let w = [0.032, 0.034, 0.032, 0.028, 0.022, 0.014, 0.005];
        add_card(b, &sp, &w, if s > 0.0 { Axis::PosX } else { Axis::NegX }, 0.005);

        let q0 = Vec3::new(s * 0.080, 1.470, -0.078);
        let q1 = Vec3::new(s * 0.072, 1.296, -0.070);
        let sq = spine_curve(q0, q1, s * 0.004, 6);
        let wq = [0.018, 0.020, 0.018, 0.014, 0.009, 0.003];
        add_card(b, &sq, &wq, Axis::NegZ, 0.004);
    }
}

fn add_back(b: &mut Builder) {
    let spine = [
        Vec3::new(0.0, 1.508, 0.006),
        Vec3::new(0.0, 1.486, 0.048),
        Vec3::new(0.0, 1.440, 0.078),
        Vec3::new(0.0, 1.372, 0.096),
        Vec3::new(0.0, 1.300, 0.104),
        Vec3::new(0.0, 1.240, 0.107),
    ];
    let w = [0.078, 0.098, 0.116, 0.126, 0.128, 0.122];
    add_card(b, &spine, &w, Axis::PosZ, 0.010);

    for s in [-1.0f32, 1.0] {
        let p0 = Vec3::new(s * 0.094, 1.470, 0.030);
        let p1 = Vec3::new(s * 0.116, 1.236, 0.086);
        let sp = spine_curve(p0, p1, s * 0.008, 6);
        let w = [0.030, 0.042, 0.050, 0.050, 0.042, 0.026];
        add_card(b, &sp, &w, if s > 0.0 { Axis::PosX } else { Axis::NegX }, 0.006);
    }
}

fn finish_normals(b: &mut Builder) {
    let mut acc = vec![Vec3::ZERO; b.verts.len()];
    for t in b.idx.chunks_exact(3) {
        let (ia, ib, ic) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let pa = Vec3::from(b.verts[ia].pos);
        let pb = Vec3::from(b.verts[ib].pos);
        let pc = Vec3::from(b.verts[ic].pos);
        let n = (pb - pa).cross(pc - pa);
        acc[ia] += n;
        acc[ib] += n;
        acc[ic] += n;
    }
    for (i, v) in b.verts.iter_mut().enumerate() {
        let n = acc[i].normalize_or_zero();
        let n = if n.dot(Vec3::from(v.pos) - C) < 0.0 { -n } else { n };
        v.normal = n.to_array();
    }
}

/// The geometry above is authored with the face at -Z; the base avatar faces
/// +Z, so mirror the finished mesh through the head's mid plane.
fn mirror_z(b: &mut Builder) {
    const CZ: f32 = 0.030;
    for v in &mut b.verts {
        v.pos[2] = 2.0 * CZ - v.pos[2];
        v.normal[2] = -v.normal[2];
    }
}

fn build_primitive() -> Primitive {
    let mut b = Builder::new();
    add_cap(&mut b);
    add_bangs(&mut b);
    add_center_part(&mut b);
    add_side_locks(&mut b);
    add_back(&mut b);
    finish_normals(&mut b);
    mirror_z(&mut b);
    Primitive {
        name: "Hair".to_string(),
        vertices: b.verts,
        indices: b.idx,
        material: 0,
        skin: None,
        skin_index: None,
        node: None,
        morph_positions: Vec::new(),
        morph_names: Vec::new(),
    }
}

/// Replace the placeholder hair primitive(s) in `scene` with the procedural
/// bob.  Returns `false` if the scene has no hair primitive to replace.
pub fn attach(scene: &mut Scene) -> bool {
    let hair: Vec<usize> = scene
        .primitives
        .iter()
        .enumerate()
        .filter(|(_, p)| p.name.to_ascii_lowercase().contains("hair"))
        .map(|(i, _)| i)
        .collect();
    let Some(&first) = hair.first() else { return false };

    let mut prim = build_primitive();
    prim.material = scene.primitives[first].material;

    // Rigidly bind the hair to the head bone so body parameters (head size,
    // height, ...) move it together with the face.
    if let Some(skin) = scene.skins.first() {
        if let Some(slot) = skin.joints.iter().position(|&n| {
            scene.nodes.get(n).map(|x| x.name == "J_Bip_C_Head").unwrap_or(false)
        }) {
            let n = prim.vertices.len();
            prim.skin = Some(vroid_model::SkinWeights {
                joints: vec![[slot as u16, 0, 0, 0]; n],
                weights: vec![[1.0, 0.0, 0.0, 0.0]; n],
            });
            prim.skin_index = Some(0);
        }
    }
    if let Some(m) = scene.materials.get_mut(prim.material) {
        m.double_sided = true;
        m.alpha_mode = AlphaMode::Opaque;
        m.mtoon.enabled = true;
        m.mtoon.shade_color = [0.60, 0.60, 0.63];
        m.mtoon.outline_width = 0.0;
        if m.base_color[..3].iter().all(|c| *c > 0.98) {
            m.base_color = [0.55, 0.38, 0.28, 1.0];
        }
    }
    scene.primitives[first] = prim;
    for &i in hair.iter().skip(1).rev() {
        scene.primitives.remove(i);
    }
    true
}
