// Fragment shader prefix.
pub const PREFIX: &str = "#version 150 core

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

// Fragment shader suffix.
pub const SUFFIX: &str = "

    void main() {
            mainImage(fragColor, fragCoord);
    }
";
pub const DEFAULT_VERT_SHADER: &str = "#version 150 core

uniform vec3 iResolution;

in vec2 position;
out vec2 fragCoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);

    // Convert from OpenGL coordinate system (with origin in center
    // of screen) to Shadertoy/texture coordinate system (with origin
    // in lower left corner)
    fragCoord = (gl_Position.xy + vec2(1.0)) / vec2(2.0) * iResolution.xy;
}
";

pub const TEST_MOUSE_SHADER: &str = "
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

pub const DEFAULT_FRAG_SHADER: &str = "
    /* This animation is the material of my first youtube tutorial about creative 
   coding, which is a video in which I try to introduce programmers to GLSL 
   and to the wonderful world of shaders, while also trying to share my recent 
   passion for this community.
                                       Video URL: https://youtu.be/f4s1h2YETNY
*/

//https://iquilezles.org/articles/palettes/
vec3 palette( float t ) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263,0.416,0.557);

    return a + b*cos( 6.28318*(c*t+d) );
}

//https://www.shadertoy.com/view/mtyGWy
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    vec2 uv0 = uv;
    vec3 finalColor = vec3(0.0);
    
    for (float i = 0.0; i < 4.0; i++) {
        uv = fract(uv * 1.5) - 0.5;

        float d = length(uv) * exp(-length(uv0));

        vec3 col = palette(length(uv0) + i*.4 + iTime*.4);

        d = sin(d*8. + iTime)/8.;
        d = abs(d);

        d = pow(0.01 / d, 1.2);

        finalColor += col * d;
    }
        
    fragColor = vec4(finalColor, 1.0);
}
";
