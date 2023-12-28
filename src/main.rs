// #![windows_subsystem = "windows"]
#![windows_subsystem = "console"]

use std::error::Error;
use std::ffi::{c_void, CStr, CString};
use std::num::{NonZeroIsize, NonZeroU32};
use std::ops::Deref;
use std::time::Instant;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoopBuilder;
use winit::platform::windows::WindowBuilderExtWindows;
use winit::raw_window_handle as rwh_06;

use raw_window_handle::HasRawWindowHandle;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::window::WindowBuilder;

use const_cstr::const_cstr;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::SwapInterval;
use glutin_winit::{self, DisplayBuilder, GlWindow};

use winapi::shared::windef::{HWND, RECT};
use winapi::um::winuser::GetClientRect;

pub mod gl {
    #![allow(clippy::all)]
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/gl_bindings.rs"));

    pub use Gles2 as Gl;
}

enum ScreensaverMode {
    Settings(bool),
    Child(rwh_06::Win32WindowHandle),
    Run,
}
fn parse_args<T>(mut iter: T) -> ScreensaverMode
where
    T: Iterator<Item = String>,
{
    let _ = iter.next();
    match (dbg!(iter.next().as_deref()), dbg!(iter.next().as_deref())) {
        (None, None) => ScreensaverMode::Settings(false),
        (Some("/c"), None) => ScreensaverMode::Settings(true),
        (Some("/p"), Some(hwnd_str)) => {
            let handle = unsafe {
                rwh_06::Win32WindowHandle::new(NonZeroIsize::new_unchecked(
                    hwnd_str.parse::<isize>().unwrap(),
                ))
            };
            ScreensaverMode::Child(handle)
        }
        (Some("/s"), None) => ScreensaverMode::Run,
        _ => unimplemented!(),
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let mode = parse_args(std::env::args());
    use ScreensaverMode::*;
    let event_loop = EventLoopBuilder::new().build().unwrap();
    match mode {
        Child(parent_window) => do_main(event_loop, Some(parent_window), false),
        Run => {
            unimplemented!()
        }
        _ => unimplemented!(),
    }
}
fn gldbg(s: &str, gl: &gl::Gl) {
    unsafe {
        let err = gl.GetError();
        if err != gl::NO_ERROR {
            println!("{s} GL error: {err}");
        }
    }
}
fn do_main(
    event_loop: winit::event_loop::EventLoop<()>,
    maybe_parent: Option<rwh_06::Win32WindowHandle>,
    modal: bool,
) -> Result<(), Box<dyn Error>> {
    let maybe_client_area = maybe_parent.map(|p| {
        let mut client_rect: RECT = Default::default();
        unsafe {
            GetClientRect(
                std::mem::transmute(isize::from(p.hwnd)),
                &mut client_rect as *mut _,
            );
        }
        client_rect
    });
    // Only Windows requires the window to be present before creating the display.
    // Other platforms don't really need one.
    //
    let mut window_builder = unsafe {
        WindowBuilder::new()
            .with_transparent(true)
            .with_parent_window(maybe_parent.map(rwh_06::RawWindowHandle::Win32))
    };
    if let Some(client_area) = maybe_client_area {
        window_builder = window_builder.with_inner_size(PhysicalSize::new(
            client_area.right - client_area.left,
            client_area.bottom - client_area.top,
        ));
    }

    // The template will match only the configurations supporting rendering
    // to windows.
    let template = ConfigTemplateBuilder::new().with_alpha_size(8);

    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    let (mut window, gl_config) = display_builder.build(&event_loop, template, |configs| {
        // Find the config with the maximum number of samples, so our triangle will
        // be smooth.
        configs
            .reduce(|accum, config| {
                let transparency_check = config.supports_transparency().unwrap_or(false)
                    & !accum.supports_transparency().unwrap_or(false);

                if transparency_check || config.num_samples() > accum.num_samples() {
                    config
                } else {
                    accum
                }
            })
            .unwrap()
    })?;

    println!("Picked a config with {} samples", gl_config.num_samples());

    let (width, height) = dbg!(window
        .as_ref()
        .map(|window| {
            let sz = window.inner_size();
            window.set_decorations(false);
            (sz.width, sz.height)
        })
        .unwrap_or((0, 0)));
    let window_handle = window.as_ref().map(|window| window.raw_window_handle());

    // XXX The display could be obtained from any object created by it, so we can
    // query it from the config.
    let gl_display = gl_config.display();

    // The context creation part. It can be created before surface and that's how
    // it's expected in multithreaded + multiwindow operation mode, since you
    // can send NotCurrentContext, but not Surface.
    let context_attributes = ContextAttributesBuilder::new().build(window_handle);

    // Since glutin by default tries to create OpenGL core context, which may not be
    // present we should try gles.
    let fallback_context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::Gles(None))
        .build(window_handle);

    // There are also some old devices that support neither modern OpenGL nor GLES.
    // To support these we can try and create a 2.1 context.
    let legacy_context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
        .build(window_handle);

    let mut not_current_gl_context = Some(unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .unwrap_or_else(|_| {
                gl_display
                    .create_context(&gl_config, &fallback_context_attributes)
                    .unwrap_or_else(|_| {
                        gl_display
                            .create_context(&gl_config, &legacy_context_attributes)
                            .expect("failed to create context")
                    })
            })
    });

    let mut state = None;
    let mut renderer = None;

    let mut last_down_coords = None;
    let mut last_clicked_coords = None;
    let (mut mx, mut my) = (0.0, 0.0);
    let (mut is_down, mut is_clicked) = (false, false);

    event_loop.run(move |event, window_target| {
        match event {
            Event::Resumed => {
                #[cfg(android_platform)]
                println!("Android window available");

                let window = window.take().unwrap_or_else(|| {
                    let window_builder = WindowBuilder::new().with_transparent(true);
                    glutin_winit::finalize_window(window_target, window_builder, &gl_config)
                        .unwrap()
                });

                let attrs = window.build_surface_attributes(Default::default());
                let gl_surface = unsafe {
                    gl_config
                        .display()
                        .create_window_surface(&gl_config, &attrs)
                        .unwrap()
                };
                let (width, height) = (gl_surface.width().unwrap(), gl_surface.height().unwrap());
                // Make it current.
                let gl_context = not_current_gl_context
                    .take()
                    .unwrap()
                    .make_current(&gl_surface)
                    .unwrap();

                // The context needs to be current for the Renderer to set up shaders and
                // buffers. It also performs function loading, which needs a current context on
                // WGL.
                renderer.get_or_insert_with(|| {
                    Renderer::new(
                        &gl_display,
                        DEFAULT_VERT_SRC_STR,
                        // &(PREFIX.to_string() + TEST_FRAG_SHADER + SUFFIX),
                        &(PREFIX.to_string() + DEFAULT_FRAG_SRC_STR + SUFFIX),
                        dbg!(width as f32),
                        dbg!(height as f32),
                    )
                });

                // Try setting vsync.
                if let Err(res) = gl_surface
                    .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
                {
                    eprintln!("Error setting vsync: {res:?}");
                }

                assert!(state.replace((gl_context, gl_surface, window)).is_none());
            }
            Event::Suspended => {
                // This event is only raised on Android, where the backing NativeWindow for a GL
                // Surface can appear and disappear at any moment.
                println!("Android window removed");

                // Destroy the GL Surface and un-current the GL Context before ndk-glue releases
                // the window back to the system.
                let (gl_context, ..) = state.take().unwrap();
                assert!(not_current_gl_context
                    .replace(gl_context.make_not_current().unwrap())
                    .is_none());
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        // Some platforms like EGL require resizing GL surface to update the size
                        // Notable platforms here are Wayland and macOS, other don't require it
                        // and the function is no-op, but it's wise to resize it for portability
                        // reasons.
                        if let Some((gl_context, gl_surface, _)) = &state {
                            gl_surface.resize(
                                gl_context,
                                NonZeroU32::new(size.width).unwrap(),
                                NonZeroU32::new(size.height).unwrap(),
                            );
                            let renderer = renderer.as_mut().unwrap();
                            renderer.resize(size.width, size.height);
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    mx = position.x as f32;
                    my = height as f32 - position.y as f32;
                    // Flip y-axis.
                    if is_down {
                        last_down_coords = Some((mx, my))
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        if dbg!(state) == ElementState::Pressed {
                            last_down_coords = Some((mx, my));
                            last_clicked_coords = Some((mx, my));
                            is_clicked = true;
                            is_down = true;
                        } else {
                            is_down = false;
                        }
                    }
                }

                WindowEvent::CloseRequested => window_target.exit(),
                _ => (),
            },
            Event::AboutToWait => {
                if let Some((gl_context, gl_surface, window)) = &state {
                    let renderer = renderer.as_mut().unwrap();
                    renderer.incr_frame();
                    if is_down || is_clicked || last_down_coords.is_some() {
                        renderer.draw(
                            dbg!(last_down_coords),
                            dbg!(last_clicked_coords),
                            dbg!(is_down),
                            dbg!(is_clicked),
                        );
                    } else {
                        renderer.draw(last_down_coords, last_clicked_coords, is_down, is_clicked);
                    }
                    window.request_redraw();

                    gl_surface.swap_buffers(gl_context).unwrap();
                }
                is_clicked = false;
            }
            _ => (),
        }
    })?;

    Ok(())
}

fn set_uniform1f(gl: &gl::Gl, program: u32, name: *const i8, val: f32) {
    unsafe {
        let loc = gl.GetUniformLocation(program, name);
        if loc < 0 {
            return;
        }
        gl.Uniform1f(loc, val)
    }
}
fn set_uniform1i(gl: &gl::Gl, program: u32, name: *const i8, val: i32) {
    unsafe {
        let loc = gl.GetUniformLocation(program, name);
        if loc < 0 {
            return;
        }
        gl.Uniform1i(loc, val)
    }
}
fn set_uniform3f(gl: &gl::Gl, program: u32, name: *const i8, x: f32, y: f32, z: f32) {
    unsafe {
        let loc = gl.GetUniformLocation(program, name);
        if loc < 0 {
            return;
        }
        gl.Uniform3f(loc, x, y, z)
    }
}

fn set_uniform4f(gl: &gl::Gl, program: u32, name: *const i8, x: f32, y: f32, z: f32, w: f32) {
    unsafe {
        let loc = gl.GetUniformLocation(program, name);
        if loc < 0 {
            return;
        }
        gl.Uniform4f(loc, x, y, z, w)
    }
}

fn print_log(v: &[i8]) {
    unsafe {
        let s = String::from_utf8_lossy(std::mem::transmute(v));
        println!("{}", s);
    }
}

pub struct Renderer {
    program: gl::types::GLuint,
    vao: gl::types::GLuint,
    vbo: gl::types::GLuint,
    gl: gl::Gl,

    start_time: Instant,
    width: f32,
    height: f32,
    time: f32,
    frame: i32,
}

impl Renderer {
    pub fn new<D: GlDisplay>(
        gl_display: &D,
        vertex_source: &str,
        fragment_source: &str,
        width: f32,
        height: f32,
    ) -> Self {
        unsafe {
            let gl = gl::Gl::load_with(|symbol| {
                let symbol = CString::new(symbol).unwrap();
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            });

            if let Some(renderer) = get_gl_string(&gl, gl::RENDERER) {
                println!("Running on {}", renderer.to_string_lossy());
            }
            if let Some(version) = get_gl_string(&gl, gl::VERSION) {
                println!("OpenGL Version {}", version.to_string_lossy());
            }

            if let Some(shaders_version) = get_gl_string(&gl, gl::SHADING_LANGUAGE_VERSION) {
                println!("Shaders version on {}", shaders_version.to_string_lossy());
            }

            // println!("{vertex_source} {fragment_source}");
            let vertex_shader = create_shader(&gl, gl::VERTEX_SHADER, vertex_source.as_bytes());
            let fragment_shader =
                create_shader(&gl, gl::FRAGMENT_SHADER, fragment_source.as_bytes());

            let program = gl.CreateProgram();
            gldbg("CreateProgram", &gl);

            gl.AttachShader(program, fragment_shader);
            gldbg("AttachShader2", &gl);
            gl.AttachShader(program, vertex_shader);
            gldbg("AttachShader1", &gl);

            gl.LinkProgram(program);
            gldbg("LinkProgram", &gl);

            let mut status = 0;
            gl.GetProgramiv(program, gl::LINK_STATUS, &mut status);
            if status == (gl::TRUE as i32) {
                println!("== LINK SUCCEEDED");
            } else {
                println!("== LINK FAILED");
                let mut log_len: i32 = 0;
                gl.GetProgramiv(program, gl::INFO_LOG_LENGTH, dbg!(&mut log_len) as *mut _);
                let mut log_data: Vec<i8> = vec![0; log_len as usize];
                gl.GetProgramInfoLog(program, log_len, &mut log_len, log_data.as_mut_ptr());
                print_log(&log_data);
            }
            gl.UseProgram(program);
            gldbg("UseProgram", &gl);

            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);

            let mut vao = std::mem::zeroed();
            gl.GenVertexArrays(1, &mut vao);
            gl.BindVertexArray(vao);

            let mut vbo = std::mem::zeroed();
            gl.GenBuffers(1, &mut vbo);
            gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                (VERTEX_DATA.len() * std::mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                VERTEX_DATA.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            let pos_attrib = gl.GetAttribLocation(program, b"position\0".as_ptr() as *const _);
            gl.VertexAttribPointer(
                pos_attrib as gl::types::GLuint,
                2,
                gl::FLOAT,
                0,
                2 * std::mem::size_of::<f32>() as gl::types::GLsizei,
                std::ptr::null(),
            );
            gl.EnableVertexAttribArray(pos_attrib as gl::types::GLuint);

            Self {
                program,
                vao,
                vbo,
                gl,
                start_time: Instant::now(),
                width,
                height,
                time: 0.0,
                frame: 0,
            }
        }
    }

    pub fn incr_frame(&mut self) {
        let elapsed = self.start_time.elapsed();
        let elapsed_ms = (elapsed.as_secs() * 1000) + u64::from(elapsed.subsec_millis());
        let elapsed_sec = (elapsed_ms as f32) / 1000.0;
        self.time = elapsed_sec;
        self.frame += 1;
    }

    pub fn draw(
        &self,
        last_down_coords: Option<(f32, f32)>,
        last_clicked_coords: Option<(f32, f32)>,
        is_down: bool,
        is_clicked: bool,
    ) {
        let ldc = last_down_coords.unwrap_or((0.0, 0.0));
        let lcc = last_clicked_coords.unwrap_or((0.0, 0.0));
        let down_flag = if is_down { 1.0 } else { -1.0 };
        let clicked_flag = if is_clicked { 1.0 } else { -1.0 };
        unsafe {
            self.gl.UseProgram(self.program);
            set_uniform1f(
                &self.gl,
                self.program,
                UNIFORM_IGLOBALTIME.as_ptr(),
                self.time,
            );
            set_uniform1f(&self.gl, self.program, UNIFORM_ITIME.as_ptr(), self.time);
            set_uniform3f(
                &self.gl,
                self.program,
                UNIFORM_IRESOLUTION.as_ptr(),
                self.width,
                self.height,
                self.width / self.height,
            );
            set_uniform4f(
                &self.gl,
                self.program,
                UNIFORM_IMOUSE.as_ptr(),
                lcc.0,
                lcc.1,
                ldc.0 * down_flag,
                ldc.1 * clicked_flag,
            );
            set_uniform1i(&self.gl, self.program, UNIFORM_IFRAME.as_ptr(), self.frame);
            // dbg!(gl.GetUniformLocation(program, UNIFORM_ICHANNEL0.as_ptr()));
            // dbg!(gl.GetUniformLocation(program, UNIFORM_ICHANNEL1.as_ptr()));
            // dbg!(gl.GetUniformLocation(program, UNIFORM_ICHANNEL2.as_ptr()));
            // dbg!(gl.GetUniformLocation(program, UNIFORM_ICHANNEL3.as_ptr()));

            self.gl.BindVertexArray(self.vao);
            self.gl.BindBuffer(gl::ARRAY_BUFFER, self.vbo);

            self.gl.ClearColor(0.1, 0.1, 0.1, 0.9);
            self.gl.Clear(gl::COLOR_BUFFER_BIT);
            self.gl.DrawArrays(gl::TRIANGLES, 0, 6);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        unsafe {
            self.gl.Viewport(0, 0, width as i32, height as i32);
            self.width = width as f32;
            self.height = height as f32;
        }
    }
}

impl Deref for Renderer {
    type Target = gl::Gl;

    fn deref(&self) -> &Self::Target {
        &self.gl
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.DeleteProgram(self.program);
            self.gl.DeleteBuffers(1, &self.vbo);
            self.gl.DeleteVertexArrays(1, &self.vao);
        }
    }
}

unsafe fn create_shader(
    gl: &gl::Gl,
    shader_type: gl::types::GLenum,
    source: &[u8],
) -> gl::types::GLuint {
    let shader = gl.CreateShader(shader_type);
    gl.ShaderSource(
        shader,
        1,
        [source.as_ptr().cast()].as_ptr(),
        std::ptr::null(),
    );
    gl.CompileShader(shader);
    gldbg("CompileShader", &gl);
    let mut status = 0;
    gl.GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);
    if status == gl::TRUE as i32 {
        println!("== SHADER COMPILED");
    } else {
        let mut log_len: i32 = 0;
        gl.GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut log_len as *mut _);
        let mut log_data: Vec<i8> = vec![0; log_len as usize];
        gl.GetShaderInfoLog(shader, log_len, &mut log_len, log_data.as_mut_ptr());
        println!("== SHADER ERROR");
        print_log(&log_data);
    }
    shader
}

fn get_gl_string(gl: &gl::Gl, variant: gl::types::GLenum) -> Option<&'static CStr> {
    unsafe {
        let s = gl.GetString(variant);
        (!s.is_null()).then(|| CStr::from_ptr(s.cast()))
    }
}

#[rustfmt::skip]
static VERTEX_DATA: [f32; 12] = [
     1.0,  1.0, // TOP RIGHT
    -1.0,  1.0, // TOP LEFT
    -1.0, -1.0, // BOTTOM LEFT

     1.0,  1.0, // TOP RIGHT
    -1.0, -1.0, // BOTTOM LEFT
     1.0, -1.0, // BOTTOM RIGHT
];

// Default shaders.
pub static DEFAULT_VERT_SRC_STR: &str = include_str!("../shaders/default.vert");
pub static DEFAULT_FRAG_SRC_STR: &str = include_str!("../shaders/default.frag");

// Fragment shader prefix.
const PREFIX: &str = "
    #version 150 core

    uniform float     iGlobalTime;
    uniform float     iTime;
    uniform vec3      iResolution;
    uniform vec4      iMouse;
    uniform int       iFrame;
    uniform sampler2D iChannel0;
    uniform sampler2D iChannel1;
    uniform sampler2D iChannel2;
    uniform sampler2D iChannel3;

    in vec2 fragCoord;
    out vec4 fragColor;


";

const_cstr! {
    UNIFORM_IGLOBALTIME = "iGlobalTime";
    UNIFORM_ITIME = "iTime";
    UNIFORM_IRESOLUTION = "iResolution";
    UNIFORM_IMOUSE = "iMouse";
    UNIFORM_IFRAME = "iFrame";
    UNIFORM_ICHANNEL0 = "iChannel0";
    UNIFORM_ICHANNEL1 = "iChannel1";
    UNIFORM_ICHANNEL2 = "iChannel2";
    UNIFORM_ICHANNEL3 = "iChannel3";
}

const TEST_FRAG_SHADER: &str = "
// Created by inigo quilez - iq/2013
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/

// Shows how to use the mouse input (only left button supported):
//
//      mouse.xy  = mouse position during last button down
//  abs(mouse.zw) = mouse position during last button click
// sign(mouze.z)  = button is down
// sign(mouze.w)  = button is clicked

float distanceToSegment( vec2 a, vec2 b, vec2 p )
{
        vec2 pa = p - a, ba = b - a;
        float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
        return length( pa - ba*h );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
        vec2 p = fragCoord / iResolution.x;
    vec2 cen = 0.5*iResolution.xy/iResolution.x;
    vec4 m = iMouse / iResolution.x;

        vec3 col = vec3(0.0);

        if( m.z>0.0 ) // button is down
        {
                float d = distanceToSegment( m.xy, abs(m.zw), p );
        col = mix( col, vec3(1.0,1.0,0.0), 1.0-smoothstep(.004,0.008, d) );
        }
        if( m.w>0.0 ) // button click
        {
        col = mix( col, vec3(1.0,1.0,1.0), 1.0-smoothstep(0.1,0.105, length(p-cen)) );
    }

        col = mix( col, vec3(1.0,0.0,0.0), 1.0-smoothstep(0.03,0.035, length(p-    m.xy )) );
    col = mix( col, vec3(0.0,0.0,1.0), 1.0-smoothstep(0.03,0.035, length(p-abs(m.zw))) );

        fragColor = vec4( col, 1.0 );
}
";
// Fragment shader suffix.
const SUFFIX: &str = "

    void main() {
            mainImage(fragColor, fragCoord);
    }
";
