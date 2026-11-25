use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use vroid_avatar::{AvatarDoc, MaterialOverride, parse_weights};
use vroid_model::Scene;
use vroid_render::{RenderParams, Renderer};

#[derive(Parser)]
#[command(name = "vroid-rs", version, about = "Headless VRM/glTF toon renderer")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print scene statistics and material list
    Info { model: PathBuf },
    /// Render a single view to PNG
    Render(RenderArgs),
    /// Render a turntable sheet of N views into one PNG
    Sheet(SheetArgs),
    /// Render a grid of expression presets
    Expr(ExprArgs),
    /// Serve an interactive web editor on localhost
    Serve(ServeArgs),
}

#[derive(Args)]
struct ServeArgs {
    model: PathBuf,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 640)]
    width: u32,
    #[arg(long, default_value_t = 800)]
    height: u32,
}

#[derive(Args)]
struct RenderArgs {
    model: PathBuf,
    #[arg(short, long, default_value = "out.png")]
    out: PathBuf,
    /// WIDTHxHEIGHT, e.g. 1024x1024
    #[arg(long, default_value = "1024x1024")]
    size: String,
    #[arg(long, default_value_t = 20.0)]
    yaw: f32,
    #[arg(long, default_value_t = 8.0)]
    pitch: f32,
    #[arg(long, default_value_t = 28.0)]
    fov: f32,
    #[arg(long, default_value_t = 1.0)]
    zoom: f32,
    #[arg(long, default_value_t = 4)]
    samples: u32,
    #[arg(long, default_value_t = 1.2)]
    outline: f32,
    #[arg(long, default_value = "171a21")]
    bg: String,
    #[arg(long, default_value_t = -35.0)]
    light_yaw: f32,
    #[arg(long, default_value_t = 40.0)]
    light_pitch: f32,
    /// Camera target "x,y,z" in model space (default: scene center)
    #[arg(long)]
    target: Option<String>,
    /// Override the auto-fit radius (close-ups)
    #[arg(long)]
    radius: Option<f32>,
    /// Expression presets, e.g. happy=1,blink=0.4
    #[arg(long)]
    expr: Option<String>,
    /// Raw morph weights, e.g. ALL_Joy=0.8 (repeatable, comma separated)
    #[arg(long)]
    morph: Option<String>,
    /// Avatar document (JSON) with expressions/morphs/material overrides
    #[arg(long)]
    doc: Option<PathBuf>,
    /// Write the resulting avatar document to this path
    #[arg(long)]
    save_doc: Option<PathBuf>,
    /// List morph targets of the loaded model and exit
    #[arg(long)]
    list_morphs: bool,
    /// Skip hair primitives
    #[arg(long)]
    no_hair: bool,
    // Body sliders, each in [-1, 1].
    #[arg(long, default_value_t = 0.0)] height: f32,
    #[arg(long, default_value_t = 0.0)] head: f32,
    #[arg(long, default_value_t = 0.0)] neck: f32,
    #[arg(long, default_value_t = 0.0)] arms: f32,
    #[arg(long, default_value_t = 0.0)] legs: f32,
    #[arg(long, default_value_t = 0.0)] shoulders: f32,
    #[arg(long, default_value_t = 0.0)] hips: f32,
    #[arg(long, default_value_t = 0.0)] waist: f32,
    #[arg(long, default_value_t = 0.0)] bust: f32,
    #[arg(long, default_value_t = 0.0)] thickness: f32,
}

#[derive(Args)]
struct ExprArgs {
    model: PathBuf,
    #[arg(short, long, default_value = "expressions.png")]
    out: PathBuf,
    #[arg(long, default_value_t = 320)]
    size: u32,
    #[arg(long, default_value_t = 1.6)]
    zoom: f32,
    /// Camera target "x,y,z" (default: model head height)
    #[arg(long)]
    target: Option<String>,
    /// Comma separated preset names (default: all except lookAt)
    #[arg(long)]
    only: Option<String>,
    /// Skip hair primitives
    #[arg(long)]
    no_hair: bool,
}

#[derive(Args)]
struct SheetArgs {
    model: PathBuf,
    #[arg(short, long, default_value = "sheet.png")]
    out: PathBuf,
    #[arg(long, default_value_t = 4)]
    views: u32,
    #[arg(long, default_value_t = 640)]
    size: u32,
    #[arg(long, default_value_t = 10.0)]
    pitch: f32,
    #[arg(long, default_value_t = 1.2)]
    outline: f32,
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s.split_once('x').context("size must look like 1024x1024")?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

fn parse_hex(s: &str) -> Result<[f32; 3]> {
    let h = s.trim_start_matches('#');
    if h.len() != 6 {
        anyhow::bail!("expected 6 hex digits, got {s}");
    }
    let v = u32::from_str_radix(h, 16)?;
    Ok([
        ((v >> 16) & 0xff) as f32 / 255.0,
        ((v >> 8) & 0xff) as f32 / 255.0,
        (v & 0xff) as f32 / 255.0,
    ])
}

fn print_info(scene: &Scene, renderer: Option<&Renderer>) {
    println!("scene      : {}", scene.name);
    println!("generator  : {}", scene.generator);
    if let Some(v) = &scene.meta.vrm_version {
        println!("vrm        : {v}");
    }
    if let Some(n) = &scene.meta.name {
        println!("meta name  : {n}");
    }
    if !scene.meta.authors.is_empty() {
        println!("authors    : {}", scene.meta.authors.join(", "));
    }
    println!(
        "geometry   : {} primitives, {} vertices, {} triangles",
        scene.primitives.len(),
        scene.vertex_count(),
        scene.triangle_count()
    );
    println!(
        "bounds     : min {:?}  max {:?}",
        scene.bounds_min, scene.bounds_max
    );
    println!(
        "size       : {:.3} x {:.3} x {:.3} (radius {:.3})",
        scene.bounds_max.x - scene.bounds_min.x,
        scene.bounds_max.y - scene.bounds_min.y,
        scene.bounds_max.z - scene.bounds_min.z,
        scene.radius()
    );
    println!("materials  : {}  textures: {}  skins: {}", scene.materials.len(), scene.textures.len(), scene.skins.len());
    for (i, m) in scene.materials.iter().enumerate() {
        println!(
            "  [{i}] {:<28} mtoon={:<5} tex={:<3} alpha={:?} double={}",
            m.name,
            m.mtoon.enabled,
            m.base_color_tex.map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
            m.alpha_mode,
            m.double_sided
        );
    }
    for p in &scene.primitives {
        println!(
            "  prim {:<28} mat={} verts={} tris={} skin={} morphs={}",
            p.name,
            p.material,
            p.vertices.len(),
            p.indices.len() / 3,
            p.skin.is_some(),
            p.morph_positions.len()
        );
    }
    if let Some(r) = renderer {
        println!("gpu        : {}", r.adapter_name());
    }
}

fn build_doc(a: &RenderArgs) -> Result<AvatarDoc> {
    let mut doc = match &a.doc {
        Some(p) => AvatarDoc::load(p)?,
        None => AvatarDoc::default(),
    };
    if let Some(e) = &a.expr {
        for (k, v) in parse_weights(e)? {
            doc.expressions.insert(k, v);
        }
    }
    if let Some(m) = &a.morph {
        for (k, v) in parse_weights(m)? {
            doc.morphs.insert(k, v);
        }
    }
    doc.body = vroid_avatar::BodyParams {
        height: a.height,
        head_size: a.head,
        neck: a.neck,
        arms: a.arms,
        legs: a.legs,
        shoulders: a.shoulders,
        hips: a.hips,
        waist: a.waist,
        bust: a.bust,
        thickness: a.thickness,
    };
    Ok(doc)
}

fn parse_target(t: &Option<String>) -> Result<Option<[f32; 3]>> {
    match t {
        Some(t) => {
            let v: Vec<f32> = t.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()?;
            if v.len() != 3 {
                anyhow::bail!("--target needs three numbers");
            }
            Ok(Some([v[0], v[1], v[2]]))
        }
        None => Ok(None),
    }
}

fn save_png(path: &PathBuf, rgba: &[u8], w: u32, h: u32) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    image::save_buffer(path, rgba, w, h, image::ExtendedColorType::Rgba8)
        .with_context(|| format!("saving {}", path.display()))
}

/// Load a model and replace its placeholder hair with the procedural bob.
fn load_avatar(path: &std::path::Path) -> Result<Scene> {
    let mut scene = vroid_model::load(path)?;
    vroid_avatar::hair::attach(&mut scene);
    Ok(scene)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { model } => {
            let scene = vroid_model::load(&model)?;
            let renderer = Renderer::new().ok();
            print_info(&scene, renderer.as_ref());
        }
        Cmd::Render(a) => {
            let mut scene = load_avatar(&a.model)?;
            let (w, h) = parse_size(&a.size)?;
            if a.list_morphs {
                for p in &scene.primitives {
                    for (i, n) in p.morph_names.iter().enumerate() {
                        println!("{i:3} {n}");
                    }
                }
                return Ok(());
            }
            let mut doc = build_doc(&a)?;
            if let Some(p) = &a.save_doc {
                doc.model = a.model.display().to_string();
                doc.save(p)?;
            }
            doc.apply(&mut scene);
            if a.no_hair {
                scene.primitives.retain(|p| !p.name.to_lowercase().contains("hair"));
            }
            print_info(&scene, None);
            let renderer = Renderer::new()?;
            println!("gpu        : {}", renderer.adapter_name());
            let p = RenderParams {
                width: w,
                height: h,
                yaw_deg: a.yaw,
                pitch_deg: a.pitch,
                fov_deg: a.fov,
                zoom: a.zoom,
                samples: a.samples,
                outline_px: a.outline,
                bg: parse_hex(&a.bg)?,
                light_yaw_deg: a.light_yaw,
                light_pitch_deg: a.light_pitch,
                target: parse_target(&a.target)?,
                radius: a.radius,
            };
            let t0 = std::time::Instant::now();
            let rgba = renderer.render(&scene, &p)?;
            save_png(&a.out, &rgba, w, h)?;
            println!("rendered   : {} ({w}x{h}, {:.2}s)", a.out.display(), t0.elapsed().as_secs_f32());
        }
        Cmd::Sheet(a) => {
            let scene = load_avatar(&a.model)?;
            let renderer = Renderer::new()?;
            let views = a.views.max(1);
            let size = a.size.max(64);
            let mut sheet = image::RgbaImage::new(size * views, size);
            for i in 0..views {
                let yaw = 360.0 * i as f32 / views as f32;
                let p = RenderParams {
                    width: size,
                    height: size,
                    yaw_deg: yaw,
                    pitch_deg: a.pitch,
                    samples: 4,
                    outline_px: a.outline,
                    ..Default::default()
                };
                let rgba = renderer.render(&scene, &p)?;
                let img = image::RgbaImage::from_raw(size, size, rgba).context("frame size mismatch")?;
                image::imageops::replace(&mut sheet, &img, (size * i) as i64, 0);
                println!("view {}/{views} yaw={yaw:.0}", i + 1);
            }
            sheet.save(&a.out)?;
            println!("sheet      : {} ({size}x{size} x{views})", a.out.display());
        }
        Cmd::Serve(a) => {
            serve(a)?;
        }
        Cmd::Expr(a) => {
            let mut base = load_avatar(&a.model)?;
            if a.no_hair {
                base.primitives.retain(|p| !p.name.to_lowercase().contains("hair"));
            }
            let renderer = Renderer::new()?;
            let size = a.size.max(64);
            let target = match parse_target(&a.target)? {
                Some(t) => Some(t),
                None => Some([0.0, base.bounds_max.y * 0.9, 0.03]),
            };
            let names: Vec<String> = match &a.only {
                Some(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
                None => vroid_avatar::expression_names().filter(|n| *n != "lookAt").map(str::to_string).collect(),
            };
            let cols = (names.len() as f32).sqrt().ceil() as u32;
            let cols = cols.max(1);
            let rows = ((names.len() as u32) + cols - 1) / cols;
            let mut sheet = image::RgbaImage::new(size * cols, size * rows);
            for (i, name) in names.iter().enumerate() {
                let mut scene = base.clone();
                let mut doc = AvatarDoc::default();
                doc.expressions.insert(name.clone(), 1.0);
                doc.apply(&mut scene);
                let p = RenderParams {
                    width: size,
                    height: size,
                    yaw_deg: 0.0,
                    pitch_deg: 0.0,
                    samples: 4,
                    zoom: a.zoom,
                    target,
                    radius: Some(0.15),
                    ..Default::default()
                };
                let rgba = renderer.render(&scene, &p)?;
                let img = image::RgbaImage::from_raw(size, size, rgba).context("frame size mismatch")?;
                let (cx, cy) = ((i as u32 % cols) * size, (i as u32 / cols) * size);
                image::imageops::replace(&mut sheet, &img, cx as i64, cy as i64);
                println!("expr {}/{}: {name}", i + 1, names.len());
            }
            sheet.save(&a.out)?;
            println!("expressions: {} ({}x{} grid)", a.out.display(), cols, rows);
        }
    }
    Ok(())
}


// ------------------------------------------------------------------ editor server

static EDITOR_HTML: &str = include_str!("editor.html");

fn hex_rgba(h: &str, a: f32) -> Result<[f32; 4]> {
    let c = parse_hex(h)?;
    Ok([c[0], c[1], c[2], a])
}

fn scale3(c: [f32; 4], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let hi = (b[i + 1] as char).to_digit(16);
                let lo = (b[i + 2] as char).to_digit(16);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                    continue;
                }
                out.push(b[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_query(q: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in q.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(pct_decode(k), pct_decode(v));
    }
    map
}

fn qf(q: &HashMap<String, String>, key: &str, default: f32) -> f32 {
    q.get(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn encode_png(rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let img = image::RgbaImage::from_raw(w, h, rgba.to_vec())
        .context("rgba buffer does not match dimensions")?;
    let mut cur = std::io::Cursor::new(Vec::new());
    img.write_to(&mut cur, image::ImageFormat::Png)
        .context("encoding png")?;
    Ok(cur.into_inner())
}

fn respond(stream: &mut std::net::TcpStream, status: u16, ctype: &str, body: &[u8]) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn render_query(
    base: &Scene,
    renderer: &Renderer,
    q: &HashMap<String, String>,
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>> {
    let mut doc = AvatarDoc::default();

    if let Some(h) = q.get("hair").filter(|h| !h.is_empty()) {
        let c = hex_rgba(h, 1.0)?;
        let shade = scale3(c, 0.58);
        let outline = scale3(c, 0.28);
        for i in 0..6 {
            doc.materials.insert(
                format!("Hair_{i}"),
                MaterialOverride {
                    base_color: Some(c),
                    shade_color: Some(shade),
                    outline_width: None,
                    outline_color: Some(outline),
                },
            );
        }
    }
    if let Some(h) = q.get("eye").filter(|h| !h.is_empty()) {
        let c = hex_rgba(h, 1.0)?;
        doc.materials.insert(
            "F00_000_00_EyeIris_00_EYE".to_string(),
            MaterialOverride {
                base_color: Some(c),
                ..Default::default()
            },
        );
    }
    if let Some(h) = q.get("skin").filter(|h| !h.is_empty()) {
        let c = hex_rgba(h, 1.0)?;
        for name in ["F00_000_00_Face_00_SKIN", "Body_Skin"] {
            doc.materials.insert(
                name.to_string(),
                MaterialOverride {
                    base_color: Some(c),
                    ..Default::default()
                },
            );
        }
    }
    if let Some(e) = q.get("expr").filter(|e| !e.is_empty() && *e != "none") {
        let (name, w) = match e.split_once(':') {
            Some((n, w)) => (n, w.parse::<f32>().unwrap_or(1.0)),
            None => (e.as_str(), 1.0),
        };
        doc.expressions.insert(name.to_string(), w.clamp(0.0, 1.0));
    }

    doc.body = vroid_avatar::BodyParams {
        height: qf(q, "height", 0.0),
        head_size: qf(q, "head", 0.0),
        neck: qf(q, "neck", 0.0),
        arms: qf(q, "arms", 0.0),
        legs: qf(q, "legs", 0.0),
        shoulders: qf(q, "shoulders", 0.0),
        hips: qf(q, "hips", 0.0),
        waist: qf(q, "waist", 0.0),
        bust: qf(q, "bust", 0.0),
        thickness: qf(q, "thickness", 0.0),
    };

    let mut scene = base.clone();
    doc.apply(&mut scene);
    if q.get("nohair").map(|v| v == "1" || v == "true").unwrap_or(false) {
        scene.primitives.retain(|p| !p.name.to_lowercase().contains("hair"));
    }

    let (w, h) = q
        .get("size")
        .and_then(|s| parse_size(s).ok())
        .unwrap_or((dw, dh));
    let bg = q
        .get("bg")
        .map(|s| parse_hex(s))
        .transpose()?
        .unwrap_or([0.09, 0.10, 0.13]);

    let p = RenderParams {
        width: w,
        height: h,
        yaw_deg: qf(q, "yaw", 20.0),
        pitch_deg: qf(q, "pitch", 8.0),
        fov_deg: qf(q, "fov", 28.0),
        zoom: qf(q, "zoom", 1.0),
        samples: qf(q, "samples", 4.0).max(1.0) as u32,
        outline_px: qf(q, "outline", 1.2),
        bg,
        light_yaw_deg: -35.0,
        light_pitch_deg: 40.0,
        target: parse_target(&q.get("target").cloned())?,
        radius: q.get("radius").and_then(|s| s.parse().ok()),
    };
    let rgba = renderer.render(&scene, &p)?;
    encode_png(&rgba, w, h)
}

fn serve(a: ServeArgs) -> Result<()> {
    let base = load_avatar(&a.model)?;
    let renderer = Renderer::new()?;
    eprintln!("gpu    : {}", renderer.adapter_name());
    eprintln!("model  : {} ({} primitives)", a.model.display(), base.primitives.len());
    let addr = format!("{}:{}", a.host, a.port);
    let listener = TcpListener::bind(&addr).with_context(|| format!("binding {addr}"))?;
    eprintln!("editor : http://{addr}/");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
        let mut buf = [0u8; 8192];
        let n = match stream.read(&mut buf) {
            Ok(0) | Err(_) => continue,
            Ok(n) => n,
        };
        let req = String::from_utf8_lossy(&buf[..n]);
        let line = req.lines().next().unwrap_or("");
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("");
        let target = parts.next().unwrap_or("/");
        if method != "GET" {
            respond(&mut stream, 405, "text/plain; charset=utf-8", b"method not allowed");
            continue;
        }
        let (path, query) = match target.split_once('?') {
            Some((p, q)) => (p, q),
            None => (target, ""),
        };
        eprintln!("{} {} {}", method, path, if query.is_empty() { "" } else { "(query)" });
        match path {
            "/" | "/index.html" => respond(
                &mut stream,
                200,
                "text/html; charset=utf-8",
                EDITOR_HTML.as_bytes(),
            ),
            "/render" => {
                let q = parse_query(query);
                match render_query(&base, &renderer, &q, a.width, a.height) {
                    Ok(png) => respond(&mut stream, 200, "image/png", &png),
                    Err(e) => {
                        eprintln!("render error: {e:#}");
                        respond(&mut stream, 500, "text/plain; charset=utf-8", format!("{e:#}").as_bytes());
                    }
                }
            }
            _ => respond(&mut stream, 404, "text/plain; charset=utf-8", b"not found"),
        }
    }
    Ok(())
}
