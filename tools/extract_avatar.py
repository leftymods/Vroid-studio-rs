#!/usr/bin/env python3
"""Extract VRoid Studio's built-in base avatar from data.unity3d into a GLB.

v1 scope: bind-pose geometry (Body skin, Face, optional Hair), UVs, morph
targets for the face, and textures paired by name. No skinning yet.
"""

import argparse
import io
import os
import json
import math
import struct
import sys

import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

BUNDLE = "/root/vroid/setup/app/VRoidStudio_Data/data.unity3d"

# ---------------------------------------------------------------- utilities

def pname(ptr):
    try:
        if not ptr:
            return ""
        d = ptr.read()
        n = getattr(d, "m_Name", None)
        if n:
            return n
        go = getattr(d, "m_GameObject", None)
        return pname(go) if go else type(d).__name__
    except Exception:
        return ""


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat_inv(m):
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for c in range(4):
        p = max(range(c, 4), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        d = a[c][c]
        if abs(d) < 1e-12:
            return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        a[c] = [v / d for v in a[c]]
        for r in range(4):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[4:] for row in a]


def convert(x, y, z):
    return (-x, y, z)


def find_object(env, kind, pred):
    for o in env.objects:
        if o.type.name != kind:
            continue
        d = o.read()
        if pred(o, d):
            return o, d
    return None, None


def _transform_of(go):
    for c in (go.m_Components or []):
        d = c.read()
        if type(d).__name__ == "Transform":
            return d
    return None


def _parent_transform(tr):
    f = getattr(tr, "m_Father", None)
    return f.read() if (f and getattr(f, "path_id", 0)) else None


def _smr_of(go):
    for c in (go.m_Components or []):
        d = c.read()
        if type(d).__name__ == "SkinnedMeshRenderer":
            return d
    return None


def avatar_instance(env):
    """Return (body, face, hair) SMRs that belong to the same prefab root."""
    body = None
    for o in env.objects:
        if o.type.name != "SkinnedMeshRenderer":
            continue
        d = o.read()
        if (pname(d.m_Mesh) == "Body" and len(d.m_Bones or []) == 70
                and bool(d.m_Materials) and pname(d.m_Materials[0]).startswith("F00_000_00_Body")):
            body = d
            break
    if body is None:
        return None, None, None
    root = _transform_of(body.m_GameObject.read())
    if root is None:
        return body, None, None
    while _parent_transform(root) is not None:
        root = _parent_transform(root)
    found = {"face": None, "hair": None}

    def walk(tr):
        go = tr.m_GameObject.read()
        smr = _smr_of(go)
        if smr is not None:
            mesh = pname(smr.m_Mesh)
            if mesh == "Face" and found["face"] is None:
                found["face"] = smr
            elif mesh.startswith("Hair") and found["hair"] is None and len(smr.m_Materials or []) >= 2:
                found["hair"] = smr
        for ch in (tr.m_Children or []):
            walk(ch.read())

    walk(root)
    return body, found["face"], found["hair"]



# ---------------------------------------------------------------- posing

def _qmat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _trs(t, r, s):
    m = _qmat(r)
    for row in range(3):
        for col in range(3):
            m[row][col] *= (s.x, s.y, s.z)[row]
    m[0][3], m[1][3], m[2][3] = t.x, t.y, t.z
    return m


_WORLD_CACHE = {}


def _world(tr):
    go = getattr(tr, "m_GameObject", None)
    key = getattr(go, "path_id", None) or id(tr)
    if key in _WORLD_CACHE:
        return _WORLD_CACHE[key]
    m = _trs(tr.m_LocalPosition, tr.m_LocalRotation, tr.m_LocalScale)
    father = getattr(tr, "m_Father", None)
    if father and getattr(father, "path_id", 0):
        m = mat_mul(_world(father.read()), m)
    _WORLD_CACHE[key] = m
    return m


def _bp_matrix(bp):
    return [
        [bp.e00, bp.e01, bp.e02, bp.e03],
        [bp.e10, bp.e11, bp.e12, bp.e13],
        [bp.e20, bp.e21, bp.e22, bp.e23],
        [bp.e30, bp.e31, bp.e32, bp.e33],
    ]


def joint_matrices(smr, mode="preview"):
    """Joint matrices J = world(bone) * bindpose, in Unity space."""
    mesh = smr.m_Mesh.read()
    bind = mesh.m_BindPose or []
    out = []
    for i, bone_ptr in enumerate(smr.m_Bones):
        if mode == "bind" or i >= len(bind):
            out.append([[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)])
            continue
        tr = bone_ptr.read()
        out.append(mat_mul(_world(tr), _bp_matrix(bind[i])))
    return out


def skin_vertex(pos, joints, weights, mats):
    x = y = z = 0.0
    for j, w in zip(joints, weights):
        if w == 0.0:
            continue
        m = mats[int(j)]
        x += w * (m[0][0] * pos[0] + m[0][1] * pos[1] + m[0][2] * pos[2] + m[0][3])
        y += w * (m[1][0] * pos[0] + m[1][1] * pos[1] + m[1][2] * pos[2] + m[1][3])
        z += w * (m[2][0] * pos[0] + m[2][1] * pos[1] + m[2][2] * pos[2] + m[2][3])
    return (x, y, z)


def skin_normal(nrm, joints, weights, mats):
    x = y = z = 0.0
    for j, w in zip(joints, weights):
        if w == 0.0:
            continue
        m = mats[int(j)]
        x += w * (m[0][0] * nrm[0] + m[0][1] * nrm[1] + m[0][2] * nrm[2])
        y += w * (m[1][0] * nrm[0] + m[1][1] * nrm[1] + m[1][2] * nrm[2])
        z += w * (m[2][0] * nrm[0] + m[2][1] * nrm[1] + m[2][2] * nrm[2])
    L = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / L, y / L, z / L)

# ---------------------------------------------------------------- mesh data

class MeshData:
    def __init__(self, name, verts, normals, uvs, submeshes, morphs, morph_names):
        self.name = name
        self.verts = verts          # list[(x,y,z)] already converted
        self.normals = normals      # list[(x,y,z)] already converted
        self.uvs = uvs              # list[(u,v)] already converted
        self.submeshes = submeshes  # list[list[int]] triangles
        self.morphs = morphs        # list[list[(x,y,z)]] per target
        self.morph_names = morph_names


def read_mesh(mesh, keep_morphs=True, smr=None, pose="bind", rigid_bone=None):
    h = MeshHandler(mesh)
    h.process()
    if not h.m_Vertices:
        raise RuntimeError("mesh has no vertex data")
    n = len(h.m_Vertices)
    mats = None
    if smr is not None and pose == "preview":
        mats = joint_matrices(smr, "preview")
    if mats is not None and rigid_bone is not None:
        m0 = mats[rigid_bone]
        raw = [skin_vertex(v, (0,), (1.0,), (m0,)) for v in h.m_Vertices]
        nrm_raw = ([skin_normal(v[:3], (0,), (1.0,), (m0,)) for v in h.m_Normals]
                   if h.m_Normals else None)
    elif mats is not None and h.m_BoneIndices and h.m_BoneWeights:
        raw = [skin_vertex(v, h.m_BoneIndices[i], h.m_BoneWeights[i], mats)
               for i, v in enumerate(h.m_Vertices)]
        nrm_raw = ([skin_normal(v[:3], h.m_BoneIndices[i], h.m_BoneWeights[i], mats)
                    for i, v in enumerate(h.m_Normals)]
                   if h.m_Normals else None)
    else:
        raw = [tuple(v) for v in h.m_Vertices]
        nrm_raw = [tuple(v[:3]) for v in h.m_Normals] if h.m_Normals else None
    verts = [convert(*v) for v in raw]
    normals = [convert(*v) for v in nrm_raw] if nrm_raw else [(0.0, 1.0, 0.0)] * n
    uvs = [(u, 1.0 - v) for (u, v) in h.m_UV0] if h.m_UV0 else [(0.0, 0.0)] * n

    tris = h.get_triangles()
    submeshes = []
    for tris_sub in tris:
        out = []
        for tri in tris_sub:
            a, b, c = tri[0], tri[1], tri[2]
            out.extend((int(a), int(c), int(b)))
        submeshes.append(out)

    morphs = []
    names = []
    sh = mesh.m_Shapes
    if keep_morphs and sh and getattr(sh, "channels", None):
        for ci, chan in enumerate(sh.channels):
            shape = sh.shapes[ci]
            deltas = [(0.0, 0.0, 0.0)] * n
            for vi in range(shape.firstVertex, shape.firstVertex + shape.vertexCount):
                v = sh.vertices[vi]
                d = (v.vertex.x, v.vertex.y, v.vertex.z)
                if mats is not None and h.m_BoneIndices and h.m_BoneWeights:
                    d = skin_normal(d, h.m_BoneIndices[v.index], h.m_BoneWeights[v.index], mats)
                deltas[v.index] = convert(*d)
            morphs.append(deltas)
            nm = chan.name.split(".")[-1]
            names.append(nm)
    return MeshData(mesh.m_Name, verts, normals, uvs, submeshes, morphs, names)


# ---------------------------------------------------------------- textures

def texture_png(tex_obj):
    from PIL import Image

    d = tex_obj if hasattr(tex_obj, "image_data") else tex_obj.read()
    fmt = int(getattr(d, "m_TextureFormat", 0))
    if fmt == 74:  # RGBA64: 4x u16 LE per pixel, high byte gives 8-bit
        w, h = d.m_Width, d.m_Height
        rgba = d.image_data[1::2]
        img = Image.frombytes("RGBA", (w, h), rgba).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    else:
        img = tex_obj.image if hasattr(tex_obj, "image") else tex_obj.read().image
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def pick_texture(env, pattern, prefer=None):
    best = None
    for o in env.objects:
        if o.type.name != "Texture2D":
            continue
        d = o.read()
        n = getattr(d, "m_Name", "")
        if pattern not in n:
            continue
        if prefer and prefer not in n:
            continue
        if best is None or d.m_Width * d.m_Height > best[0]:
            best = (d.m_Width * d.m_Height, n, d)
    return best[1:] if best else (None, None)


# ---------------------------------------------------------------- GLB writer

class Glb:
    def __init__(self):
        self.bin = bytearray()
        self.views = []
        self.accessors = []
        self.json = {
            "asset": {"version": "2.0", "generator": "vroid-extract"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
            "images": [],
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": 0}],
        }

    def _align(self):
        while len(self.bin) % 4:
            self.bin.append(0)

    def view(self, data, target=None):
        self._align()
        off = len(self.bin)
        self.bin.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, view, comp, typ, count, mn=None, mx=None):
        a = {"bufferView": view, "componentType": comp, "count": count, "type": typ}
        if mn is not None:
            a["min"] = mn
            a["max"] = mx
        self.accessors.append(a)
        return len(self.accessors) - 1

    def add_floats(self, values, typ, comps, target=34962, bounds=False):
        flat = [c for v in values for c in v]
        data = struct.pack(f"<{len(flat)}f", *flat)
        v = self.view(data, target)
        mn = mx = None
        if bounds:
            mn = [min(v[i] for v in values) for i in range(comps)]
            mx = [max(v[i] for v in values) for i in range(comps)]
        return self.accessor(v, 5126, typ, len(values), mn, mx)

    def add_uints(self, values):
        values = [int(v) for v in values]
        data = struct.pack(f"<{len(values)}I", *values)
        v = self.view(data, 34963)
        return self.accessor(v, 5125, "SCALAR", len(values))

    def add_png(self, png):
        v = self.view(png)
        self.json["images"].append({"bufferView": v, "mimeType": "image/png"})
        idx = len(self.json["images"]) - 1
        self.json["textures"].append({"source": idx, "sampler": 0})
        return len(self.json["textures"]) - 1

    def finish(self, path):
        self._align()
        self.json["buffers"][0]["byteLength"] = len(self.bin)
        js = json.dumps(self.json, separators=(",", ":")).encode()
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(b"glTF" + struct.pack("<II", 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))


def mtoon(shade=(0.62, 0.62, 0.68), toony=0.9, outline=0.4, ocolor=(0.12, 0.10, 0.10)):
    return {
        "VRMC_materials_mtoon": {
            "specVersion": "1.0",
            "shadeColorFactor": list(shade),
            "shadingToonyFactor": toony,
            "shadingShiftFactor": 0.0,
            "rimColorFactor": [0.0, 0.0, 0.0],
            "rimLightingMixFactor": 0.0,
            "outlineWidthFactor": outline,
            "outlineColorFactor": list(ocolor),
            "transparentWithZWrite": False,
        }
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/vroid/base_female.glb")
    ap.add_argument("--hair", action="store_true", help="include hair mesh")
    ap.add_argument("--clothes", action="store_true", help="include body submeshes 1..3")
    ap.add_argument("--only", default="all", choices=["all","body","face","hair"])
    ap.add_argument("--only-prim", default="", help="comma list of face submeshes")
    ap.add_argument("--face-tex", default="/root/vroid/face_tex", help="directory with face decal PNGs")
    ap.add_argument("--face-debug", action="store_true", help="use debug.png for every face decal")
    ap.add_argument("--pose", default="bind", choices=["bind","preview"], help="bind pose or the bundle's preview pose")
    args = ap.parse_args()

    print("loading bundle...", flush=True)
    env = UnityPy.load(BUNDLE)

    body_d, face_d, hair_d = avatar_instance(env)
    if body_d is None:
        sys.exit("female body SMR not found")
    if face_d is None:
        sys.exit("female face SMR not found")

    print("body:", pname(body_d.m_GameObject), "bones", len(body_d.m_Bones))
    print("face:", pname(face_d.m_GameObject), "bones", len(face_d.m_Bones))

    body_mesh = body_d.m_Mesh.read()
    face_mesh = face_d.m_Mesh.read()
    body = read_mesh(body_mesh, smr=body_d, pose=args.pose)
    face = read_mesh(face_mesh, smr=face_d, pose=args.pose)
    print(f"body: {len(body.verts)} verts, {len(body.submeshes)} submeshes, morphs {len(body.morphs)}")
    print(f"face: {len(face.verts)} verts, {len(face.submeshes)} submeshes, morphs {len(face.morphs)}")
    print("face morphs:", ", ".join(face.morph_names[:8]), "...")

    hair = None
    if args.hair or args.only == "hair":
        hd = hair_d
        if hd is not None:
            hair = read_mesh(hd.m_Mesh.read(), keep_morphs=False, smr=hd, pose=args.pose,
                             rigid_bone=(0 if args.pose == "preview" else None))
            print(f"hair: {hd.m_Mesh.read().m_Name} {len(hair.verts)} verts, {len(hair.submeshes)} submeshes")

    # textures
    # NOTE: the *Beta-to-N00* textures are UV remap maps, not albedo. The base
    # avatar is untextured (flat MToon colors) except for the hair strand map.
    tex = {}
    if hair:
        n, t = pick_texture(env, "F00_000_Hair_00")
        if t:
            tex["hair"] = texture_png(t)
            print(f"texture hair: {n} ({t.m_Width}x{t.m_Height})")

    # face decal textures (one per material, each mesh has its own 0..1 UV space)
    FACE_KEYS = ("iris", "eyewhite", "highlight", "eyeline", "eyelash", "brow", "mouth", "face", "extra")
    face_tex = {}
    if os.path.isdir(args.face_tex):
        for k in FACE_KEYS:
            fn = "debug.png" if args.face_debug else f"{k}.png"
            fp = os.path.join(args.face_tex, fn)
            if os.path.exists(fp):
                face_tex[k] = open(fp, "rb").read()
    print("face textures:", sorted(face_tex))

    glb = Glb()
    tex_idx = {k: glb.add_png(v) for k, v in tex.items()}
    tex_idx.update({f"face_{k}": glb.add_png(v) for k, v in face_tex.items()})

    HEAD_C = (0.0, 1.43, 0.0)

    def offset_verts(verts, mm):
        if mm == 0.0:
            return verts
        cx, cy, cz = HEAD_C
        out = []
        for x, y, z in verts:
            dx, dy, dz = x - cx, y - cy, z - cz
            L = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            k = mm * 0.001
            out.append((x + dx / L * k, y + dy / L * k, z + dz / L * k))
        return out

    def scale_verts(verts, idx, sy):
        if sy == 1.0:
            return verts
        used = set(idx)
        ys = [verts[k][1] for k in used]
        cy = (min(ys) + max(ys)) * 0.5
        return [(x, cy + (y - cy) * sy, z) for (x, y, z) in verts]

    def add_mesh(md, submesh_indices, materials, name, offsets=None, yscales=None):
        prims = []
        for si in submesh_indices:
            idx = md.submeshes[si]
            if not idx:
                continue
            off = offsets.get(si, 0.0) if offsets else 0.0
            verts = scale_verts(md.verts, idx, yscales.get(si, 1.0) if yscales else 1.0)
            verts = offset_verts(verts, off)
            pos = glb.add_floats(verts, "VEC3", 3, bounds=True)
            nrm = glb.add_floats(md.normals, "VEC3", 3)
            uv = glb.add_floats(md.uvs, "VEC2", 2)
            ind = glb.add_uints(idx)
            prim = {
                "attributes": {"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv},
                "indices": ind,
                "material": materials[si],
            }
            if md.morphs:
                targets = []
                for deltas in md.morphs:
                    targets.append({"POSITION": glb.add_floats(deltas, "VEC3", 3)})
                prim["targets"] = targets
            prims.append(prim)
        mesh = {"name": name, "primitives": prims}
        if md.morphs:
            mesh["extras"] = {"targetNames": md.morph_names}
            mesh["weights"] = [0.0] * len(md.morphs)
        glb.json["meshes"].append(mesh)
        node = {"name": name, "mesh": len(glb.json["meshes"]) - 1}
        glb.json["nodes"].append(node)
        glb.json["scenes"][0]["nodes"].append(len(glb.json["nodes"]) - 1)

    def add_material(name, tex_key=None, color=(1, 1, 1, 1), alpha="OPAQUE", double=False, shade=(0.62, 0.62, 0.68), outline=0.4):
        m = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(color),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "alphaMode": alpha,
            "doubleSided": double,
            "extensions": mtoon(shade=shade, outline=outline),
        }
        if alpha == "MASK":
            m["alphaCutoff"] = 0.35
        if tex_key and tex_key in tex_idx:
            m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx[tex_key]}
        glb.json["materials"].append(m)
        return len(glb.json["materials"]) - 1

    skin = (1.0, 0.86, 0.79, 1.0)

    # Body: skin only unless --clothes
    body_mats = []
    body_mats.append(add_material("Body_Skin", None, color=skin, shade=(0.86, 0.68, 0.64), outline=0.5))
    body_sub = [0]
    if args.clothes:
        for nm in ("Tops", "Bottoms", "Shoes"):
            body_mats.append(add_material(nm, None, color=(0.85, 0.85, 0.9, 1), double=True))
            body_sub.append(len(body_mats) - 1)
    if args.only in ("all","body"):
        add_mesh(body, body_sub, body_mats, "Body")

    # Face submeshes in material order from the SMR
    fnames = [pname(m) for m in (face_d.m_Materials or [])]
    print("face materials:", fnames)
    def face_key(nm):
        low = nm.lower()
        for k in ("iris", "eyewhite", "highlight", "eyeline", "eyelash", "brow", "mouth", "extra"):
            if k in low:
                return k
        return "face"

    face_mats = []
    for nm in fnames:
        k = face_key(nm)
        key = f"face_{k}" if k in face_tex else None
        if k == "iris":
            face_mats.append(add_material(nm, key, color=(0.62, 0.40, 0.26, 1), double=True, outline=0.0))
        elif k == "highlight":
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else (1, 1, 1, 0),
                                          alpha="BLEND", double=True, outline=0.0))
        elif k == "eyewhite":
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else (0.99, 0.99, 0.99, 1),
                                          double=True, outline=0.0))
        elif k in ("eyelash", "eyeline"):
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else (0.12, 0.08, 0.07, 1),
                                          alpha="MASK" if key else "OPAQUE", double=True, outline=0.0))
        elif k == "brow":
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else (0.30, 0.18, 0.13, 1),
                                          alpha="MASK" if key else "OPAQUE", double=True, outline=0.0))
        elif k == "mouth":
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else (0.62, 0.30, 0.30, 1),
                                          alpha="MASK" if key else "OPAQUE", double=True, outline=0.0))
        elif k == "extra":
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else (1, 1, 1, 0),
                                          alpha="BLEND", double=True, outline=0.0))
        else:
            face_mats.append(add_material(nm, key, color=(1, 1, 1, 1) if key else skin,
                                          double=True, shade=(0.86, 0.68, 0.64), outline=0.3))
    if args.only in ("all","face"):
        offs = {}
        for i, nm in enumerate(fnames):
            low = nm.lower()
            if "eyewhite" in low:
                offs[i] = 1.0
            elif "iris" in low:
                offs[i] = 2.0
            elif "highlight" in low:
                offs[i] = 2.6
            elif "eyeline" in low or "eyelash" in low or "brow" in low:
                offs[i] = 3.0
            elif "mouth" in low:
                offs[i] = 1.0
            elif "extra" in low:
                offs[i] = 0.6
        sub = list(range(len(face.submeshes))) if not args.only_prim else [int(x) for x in args.only_prim.split(",")]
        ysc = {}
        for i, nm in enumerate(fnames):
            low = nm.lower()
            if "brow" in low:
                ysc[i] = 0.42
            elif "eyelash" in low:
                ysc[i] = 0.8
            elif "eyeline" in low:
                ysc[i] = 0.85
        add_mesh(face, sub, face_mats, "Face", offsets=offs, yscales=ysc)

    if hair and args.only in ("all","hair"):
        hair_mats = [add_material(f"Hair_{i}", "hair", color=(0.55, 0.38, 0.28, 1), alpha="OPAQUE", double=True, shade=(0.28, 0.18, 0.13), outline=0.6)
                     for i in range(len(hair.submeshes))]
        add_mesh(hair, list(range(len(hair.submeshes))), hair_mats, "Hair")

    glb.finish(args.out)
    size = len(glb.bin)
    print(f"wrote {args.out}: {size/1e6:.1f} MB bin, {len(glb.json['meshes'])} meshes, "
          f"{len(glb.json['materials'])} materials, {len(glb.json['images'])} images")


if __name__ == "__main__":
    main()
