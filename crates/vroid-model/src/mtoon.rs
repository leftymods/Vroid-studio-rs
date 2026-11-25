//! MToon material parameters (VRM 0.x and VRM 1.0) extracted from glTF JSON.

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct MToon {
    pub enabled: bool,
    pub shade_color: [f32; 3],
    pub shading_toony: f32,
    pub shading_shift: f32,
    pub rim_color: [f32; 3],
    pub rim_lighting_mix: f32,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
}

impl Default for MToon {
    fn default() -> Self {
        Self {
            enabled: false,
            shade_color: [0.6, 0.62, 0.7],
            shading_toony: 0.9,
            shading_shift: 0.0,
            rim_color: [0.0, 0.0, 0.0],
            rim_lighting_mix: 0.0,
            outline_width: 0.0,
            outline_color: [0.1, 0.1, 0.12],
        }
    }
}

pub fn from_gltf_json(json: &Value, material_index: usize) -> MToon {
    let ext = &json["materials"][material_index]["extensions"];

    // VRM 1.0
    if let Some(m) = ext.get("VRMC_materials_mtoon") {
        return MToon {
            enabled: true,
            shade_color: color3(&m["shadeColorFactor"]).unwrap_or([0.6, 0.62, 0.7]),
            shading_toony: f32_of(&m["shadingToonyFactor"]).unwrap_or(0.9),
            shading_shift: f32_of(&m["shadingShiftFactor"]).unwrap_or(0.0),
            rim_color: color3(&m["rimColorFactor"]).unwrap_or([0.0; 3]),
            rim_lighting_mix: f32_of(&m["rimLightingMixFactor"]).unwrap_or(0.0),
            outline_width: f32_of(&m["outlineWidthFactor"]).unwrap_or(0.0),
            outline_color: color3(&m["outlineColorFactor"]).unwrap_or([0.1, 0.1, 0.12]),
        };
    }

    // VRM 0.x
    if let Some(m) = ext.get("VRM").and_then(|v| v.get("materialProperties")) {
        return MToon {
            enabled: true,
            shade_color: color3(&m["shadeColorFactor"]).unwrap_or([0.6, 0.62, 0.7]),
            shading_toony: f32_of(&m["shadingToonyFactor"]).unwrap_or(0.9),
            shading_shift: f32_of(&m["shadingShiftFactor"]).unwrap_or(0.0),
            rim_color: color3(&m["rimColorFactor"]).unwrap_or([0.0; 3]),
            rim_lighting_mix: f32_of(&m["rimLightingMixFactor"]).unwrap_or(0.0),
            outline_width: f32_of(&m["outlineWidthFactor"]).unwrap_or(0.0),
            outline_color: color3(&m["outlineColorFactor"]).unwrap_or([0.1, 0.1, 0.12]),
        };
    }

    MToon::default()
}

fn f32_of(v: &Value) -> Option<f32> {
    v.as_f64().map(|x| x as f32)
}

fn color3(v: &Value) -> Option<[f32; 3]> {
    let a = v.as_array()?;
    if a.len() < 3 {
        return None;
    }
    Some([a[0].as_f64()? as f32, a[1].as_f64()? as f32, a[2].as_f64()? as f32])
}
