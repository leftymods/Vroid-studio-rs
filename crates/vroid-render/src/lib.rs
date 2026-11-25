//! vroid-render — a small wgpu renderer with a VRM/MToon-flavoured toon shader,
//! inverted-hull outlines, MSAA and an orbit camera. Renders offscreen to RGBA8,
//! so it works headless (CI, servers, Android userspace with lavapipe).

use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use vroid_model::{AlphaMode, Scene, Vertex};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone, Debug)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub fov_deg: f32,
    pub zoom: f32,
    pub samples: u32,
    pub outline_px: f32,
    pub bg: [f32; 3],
    pub light_yaw_deg: f32,
    pub light_pitch_deg: f32,
    pub target: Option<[f32; 3]>,
    /// Override the auto-fit radius (useful for close-ups).
    pub radius: Option<f32>,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            yaw_deg: 20.0,
            pitch_deg: 8.0,
            fov_deg: 28.0,
            zoom: 1.0,
            samples: 4,
            outline_px: 1.2,
            bg: [0.09, 0.10, 0.13],
            light_yaw_deg: -35.0,
            light_pitch_deg: 40.0,
            target: None,
            radius: None,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    camera: [f32; 4],
    light: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MatUniform {
    base_color: [f32; 4],
    shade: [f32; 4],
    rim: [f32; 4],
    outline: [f32; 4],
    flags: [f32; 4],
}

const ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: Vertex::STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    bgl_globals: wgpu::BindGroupLayout,
    bgl_tex: wgpu::BindGroupLayout,
    bgl_mat: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    main: wgpu::RenderPipeline,
    blend: wgpu::RenderPipeline,
    outline: wgpu::RenderPipeline,
    adapter_info: wgpu::AdapterInfo,
}

struct GpuPrim {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    count: u32,
    tex_bg: wgpu::BindGroup,
    mat_bg: wgpu::BindGroup,
    outline_scale: f32,
    is_blend: bool,
}

impl Renderer {
    pub fn new() -> Result<Self> {
        let mut idesc = wgpu::InstanceDescriptor::new_without_display_handle();
        idesc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        let instance = wgpu::Instance::new(idesc);
        let mut opts = wgpu::RequestAdapterOptions::default();
        opts.power_preference = wgpu::PowerPreference::HighPerformance;
        let adapter = pollster::block_on(instance.request_adapter(&opts))
            .map_err(|e| anyhow!("no usable GPU adapter: {e}"))?;
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vroid-render"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .context("requesting wgpu device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("toon.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl_globals = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bgl_tex = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bgl_mat = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("toon-layout"),
            bind_group_layouts: &[Some(&bgl_globals), Some(&bgl_tex), Some(&bgl_mat)],
            immediate_size: 0,
        });

        let make = |label: &str, entry: &str, cull: Option<wgpu::Face>, depth: wgpu::CompareFunction, blend: wgpu::BlendState, depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(depth),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 4,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if entry == "vs_outline" { "fs_outline" } else { "fs_main" }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let main = make("main", "vs_main", None, wgpu::CompareFunction::LessEqual, wgpu::BlendState::ALPHA_BLENDING, true);
        let blend = make("blend", "vs_main", None, wgpu::CompareFunction::LessEqual, wgpu::BlendState::ALPHA_BLENDING, false);
        let outline = make("outline", "vs_outline", Some(wgpu::Face::Front), wgpu::CompareFunction::Less, wgpu::BlendState::REPLACE, true);

        Ok(Self { device, queue, shader, bgl_globals, bgl_tex, bgl_mat, layout, main, blend, outline, adapter_info })
    }

    pub fn adapter_name(&self) -> String {
        format!("{} ({:?})", self.adapter_info.name, self.adapter_info.backend)
    }

    pub fn render(&self, scene: &Scene, p: &RenderParams) -> Result<Vec<u8>> {
        let device = &self.device;
        let width = p.width.max(1);
        let height = p.height.max(1);
        let samples = p.samples.clamp(1, 8);

        // --- camera -------------------------------------------------------
        let center = p.target.map(Vec3::from).unwrap_or_else(|| scene.center());
        let radius = p.radius.unwrap_or_else(|| scene.radius()).max(1e-4);
        let fov = p.fov_deg.to_radians();
        let dist = radius / (fov * 0.5).tan() * 1.12 / p.zoom.max(0.01);
        let yaw = p.yaw_deg.to_radians();
        let pitch = p.pitch_deg.to_radians();
        let dir = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos());
        let eye = center + dir * dist;
        let view = glam::camera::rh::view::look_at_mat4(eye, center, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            fov,
            width as f32 / height as f32,
            dist * 0.02,
            dist * 20.0,
        );
        let view_proj = proj * view;

        let ly = p.light_yaw_deg.to_radians();
        let lp = p.light_pitch_deg.to_radians();
        let light = Vec3::new(ly.sin() * lp.cos(), lp.sin(), ly.cos() * lp.cos()).normalize();

        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            camera: [eye.x, eye.y, eye.z, 1.0],
            light: [light.x, light.y, light.z, 0.0],
            params: [p.outline_px, height as f32, 1.0, 0.0],
        };
        let gbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let gbg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &self.bgl_globals,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: gbuf.as_entire_binding() }],
        });

        // --- pipelines must match MSAA sample count -----------------------
        let (main, blend, outline) = if samples == 4 {
            (None, None, None)
        } else {
            (Some(self.build_pipeline("vs_main", "fs_main", None, wgpu::CompareFunction::LessEqual, wgpu::BlendState::ALPHA_BLENDING, true, samples)),
             Some(self.build_pipeline("vs_main", "fs_main", None, wgpu::CompareFunction::LessEqual, wgpu::BlendState::ALPHA_BLENDING, false, samples)),
             Some(self.build_pipeline("vs_outline", "fs_outline", Some(wgpu::Face::Front), wgpu::CompareFunction::Less, wgpu::BlendState::REPLACE, true, samples)))
        };
        let main_pipe = main.as_ref().unwrap_or(&self.main);
        let blend_pipe = blend.as_ref().unwrap_or(&self.blend);
        let outline_pipe = outline.as_ref().unwrap_or(&self.outline);

        // --- per-primitive GPU resources ----------------------------------
        let white = self.white_texture();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let mut tex_cache: std::collections::HashMap<usize, wgpu::BindGroup> = std::collections::HashMap::new();

        let mut prims = Vec::with_capacity(scene.primitives.len());
        for prim in &scene.primitives {
            if prim.vertices.is_empty() || prim.indices.is_empty() {
                continue;
            }
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertices"),
                contents: bytemuck::cast_slice(&prim.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("indices"),
                contents: bytemuck::cast_slice(&prim.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            if scene.materials.is_empty() {
                continue;
            }
            let mat = &scene.materials[prim.material.min(scene.materials.len() - 1)];
            let tex_bg = if let Some(ti) = mat.base_color_tex.filter(|t| *t < scene.textures.len()) {
                tex_cache
                    .entry(ti)
                    .or_insert_with(|| self.upload_texture(&scene.textures[ti], &sampler))
                    .clone()
            } else {
                white.clone()
            };
            let alpha_mode = match mat.alpha_mode {
                AlphaMode::Opaque => 0.0,
                AlphaMode::Mask => 1.0,
                AlphaMode::Blend => 2.0,
            };
            let outline_scale = if mat.mtoon.enabled {
                mat.mtoon.outline_width
            } else {
                1.0
            };
            let mu = MatUniform {
                base_color: mat.base_color,
                shade: [mat.mtoon.shade_color[0], mat.mtoon.shade_color[1], mat.mtoon.shade_color[2], mat.mtoon.shading_toony],
                rim: [mat.mtoon.rim_color[0], mat.mtoon.rim_color[1], mat.mtoon.rim_color[2], mat.mtoon.rim_lighting_mix],
                outline: [mat.mtoon.outline_color[0], mat.mtoon.outline_color[1], mat.mtoon.outline_color[2], mat.mtoon.shading_shift],
                flags: [mat.mtoon.enabled as u8 as f32, mat.alpha_cutoff, alpha_mode, outline_scale],
            };
            let mbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&mu),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material"),
                layout: &self.bgl_mat,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: mbuf.as_entire_binding() }],
            });
            prims.push(GpuPrim { vbuf, ibuf, count: prim.indices.len() as u32, tex_bg, mat_bg, outline_scale, is_blend: matches!(mat.alpha_mode, AlphaMode::Blend) });
        }

        // --- attachments --------------------------------------------------
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let msaa = (samples > 1).then(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });
        let msaa_view = msaa.as_ref().map(|t| t.create_view(&Default::default()));
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let (view, resolve) = match &msaa_view {
                Some(v) => (v, Some(&target_view)),
                None => (&target_view, None),
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("toon"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: srgb_to_linear(p.bg[0] as f64),
                            g: srgb_to_linear(p.bg[1] as f64),
                            b: srgb_to_linear(p.bg[2] as f64),
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // outlines first (inverted hull), then the shaded pass
            pass.set_pipeline(outline_pipe);
            pass.set_bind_group(0, &gbg, &[]);
            for prim in prims.iter().filter(|p| !p.is_blend && p.outline_scale > 0.0) {
                pass.set_bind_group(1, &prim.tex_bg, &[]);
                pass.set_bind_group(2, &prim.mat_bg, &[]);
                pass.set_vertex_buffer(0, prim.vbuf.slice(..));
                pass.set_index_buffer(prim.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.count, 0, 0..1);
            }

            pass.set_pipeline(main_pipe);
            pass.set_bind_group(0, &gbg, &[]);
            for prim in prims.iter().filter(|p| !p.is_blend) {
                pass.set_bind_group(1, &prim.tex_bg, &[]);
                pass.set_bind_group(2, &prim.mat_bg, &[]);
                pass.set_vertex_buffer(0, prim.vbuf.slice(..));
                pass.set_index_buffer(prim.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.count, 0, 0..1);
            }

            pass.set_pipeline(blend_pipe);
            pass.set_bind_group(0, &gbg, &[]);
            for prim in prims.iter().filter(|p| p.is_blend) {
                pass.set_bind_group(1, &prim.tex_bg, &[]);
                pass.set_bind_group(2, &prim.mat_bg, &[]);
                pass.set_vertex_buffer(0, prim.vbuf.slice(..));
                pass.set_index_buffer(prim.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.count, 0, 0..1);
            }
        }

        let padded_row = ((width * 4 + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().context("waiting for readback")??;

        let data = slice.get_mapped_range().map_err(|e| anyhow!("mapping readback: {e}"))?;
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * padded_row) as usize;
            out.extend_from_slice(&data[start..start + (width * 4) as usize]);
        }
        drop(data);
        readback.unmap();
        Ok(out)
    }

    fn build_pipeline(
        &self,
        vs: &str,
        fs: &str,
        cull: Option<wgpu::Face>,
        depth: wgpu::CompareFunction,
        blend: wgpu::BlendState,
        depth_write: bool,
        samples: u32,
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(vs),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(depth),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn upload_texture(&self, tex: &vroid_model::Texture, sampler: &wgpu::Sampler) -> wgpu::BindGroup {
        let size = wgpu::Extent3d { width: tex.width.max(1), height: tex.height.max(1), depth_or_array_layers: 1 };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&tex.name),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let row_bytes = size.width * 4;
        let padded = ((row_bytes + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let data = if padded == row_bytes {
            tex.rgba.clone()
        } else {
            let mut buf = vec![0u8; (padded * size.height) as usize];
            for y in 0..size.height {
                let src = (y * row_bytes) as usize;
                let dst = (y * padded) as usize;
                buf[dst..dst + row_bytes as usize]
                    .copy_from_slice(&tex.rgba[src..src + row_bytes as usize]);
            }
            buf
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.height),
            },
            size,
        );
        let view = texture.create_view(&Default::default());
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture"),
            layout: &self.bgl_tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    fn white_texture(&self) -> wgpu::BindGroup {
        let tex = vroid_model::Texture { name: "white".into(), width: 1, height: 1, rgba: vec![255, 255, 255, 255] };
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.upload_texture(&tex, &sampler)
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

use wgpu::util::DeviceExt;
