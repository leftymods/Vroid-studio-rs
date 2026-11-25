struct Globals {
    view_proj: mat4x4<f32>,
    camera: vec4<f32>,
    light: vec4<f32>,
    params: vec4<f32>,
};

struct MatU {
    base_color: vec4<f32>,
    shade: vec4<f32>,
    rim: vec4<f32>,
    outline: vec4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(2) @binding(0) var<uniform> m: MatU;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    o.wpos = v.pos;
    o.nrm = v.nrm;
    o.uv = v.uv;
    return o;
}

@vertex
fn vs_outline(v: VsIn) -> VsOut {
    var o: VsOut;
    let clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    let nclip = g.view_proj * vec4<f32>(v.nrm, 0.0);
    var dir = nclip.xy;
    if (length(dir) < 1e-5) {
        dir = vec2<f32>(0.0, 1.0);
    }
    let px = g.params.x * m.flags.w;
    let ndc = normalize(dir) * (px * 2.0 / g.params.y) * clip.w;
    o.clip = vec4<f32>(clip.xy + ndc, clip.z, clip.w);
    o.wpos = v.pos;
    o.nrm = v.nrm;
    o.uv = v.uv;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    var base = textureSample(tex, samp, i.uv) * m.base_color;

    let alpha_mode = m.flags.z;
    if (alpha_mode > 0.5 && alpha_mode < 1.5) {
        if (base.a < m.flags.y) {
            discard;
        }
        base.a = 1.0;
    } else if (alpha_mode < 0.5) {
        base.a = 1.0;
    }

    let n = normalize(i.nrm);
    let l = normalize(g.light.xyz);
    let ndl = dot(n, l);

    let toony = clamp(m.shade.w, 0.0, 1.0);
    let shift = m.outline.w;
    let term = smoothstep(-0.06 + shift, 0.06 + shift, ndl);
    let lit = mix(1.0 - toony, 1.0, term);

    var rgb = base.rgb * mix(m.shade.rgb, vec3<f32>(1.0), lit);

    let vdir = normalize(g.camera.xyz - i.wpos);
    let fres = pow(1.0 - clamp(dot(n, vdir), 0.0, 1.0), 4.0);
    rgb = rgb + m.rim.rgb * fres * m.rim.a;

    let warm = vec3<f32>(1.03, 1.0, 0.97);
    rgb = rgb * warm;

    return vec4<f32>(rgb, base.a);
}

@fragment
fn fs_outline(i: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(m.outline.rgb, 1.0);
}
