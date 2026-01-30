# Ash’s OBJ Viewer
## Because sometimes you just want a rotating product shot from an `.obj` without starting a Blender pilgrimage, learning seven hotkeys for “orbit”, and accidentally becoming a 3D monk.


<img src="docs/turntable.gif" width="500">


This is a **local** OBJ viewer + **turntable renderer**.  
Dark navy + orange theme included, because grey UI is how joy dies.

There is also a **Windows EXE release** for people who would rather double-click than negotiate with Python.

---

## What This Thing Does

### View OBJs
- Load an `.obj` and preview it in a real-time OpenGL viewport
- Smooth or Flat shading
- Faces / Edges / Wireframe-only
- Edge alpha slider so your wireframe isn’t screaming at you

### Baked Lighting (Without Shader Drama)
This app “bakes” lighting into vertex colours (diffuse + specular-ish) so it looks decent even if your pyqtgraph shader options are… temperamental.

- Lighting modes:
  - Headlight (camera): light follows the camera
  - Fixed (world): light stays in world space like a stage lamp
- Light brightness slider (scales diffuse + spec contribution)
- Light diffuse slider (controls how matte-bright it feels)
- Pick model face colour + light tint colour

### Studio-ish Preset
A two-tone studio preset that tweaks the vibe (and edge colour), for the “product shot” look without the full cinematic nonsense.

### Background Picker
Pick the viewport background colour.  
Important detail: renders are framebuffer grabs, so the background colour also affects your exported MP4/GIF. No magic, no post-processing, just the raw truth.

### Turntable Render Export
- Render turntables to MP4 (H.264)
- Optionally export a GIF too
- Control:
  - Axis (x, y, z)
  - Total degrees (default 360)
  - Seconds
  - FPS
- Preview 1 frame button so you don’t render 300 frames of regret
- Render clean option: disables edges during export only (preview can stay wireframed like a CAD goblin)

---

## Installation

### Option 1: Windows EXE (Recommended if you hate Python setup)

Download the latest EXE from the **Releases** page.

No Python required.  
No virtual environments.  
No spiritual growth.

Important: you will still need **FFmpeg installed separately** for MP4 export to work. See the FFmpeg section below.

---

### Option 2: Run from Source

Dependencies:
- PyQt5
- pyqtgraph
- PyOpenGL
- numpy
- trimesh
- imageio
- imageio-ffmpeg
- Pillow

#### Install FFmpeg to your PATH

#### Install the requirements.txt file  
python -m pip install -r requirements.txt

#### Or install them separately  
python -m pip install PyQt5 pyqtgraph PyOpenGL numpy trimesh imageio imageio-ffmpeg Pillow

#### Run:  
python main.py

---

## FFmpeg (Required for MP4 Export)

MP4 rendering relies on FFmpeg.  
Even if you’re using the EXE, **FFmpeg must be installed separately and available on your PATH**.

If FFmpeg is missing:
- The app will still run
- Preview will still work
- MP4 export will quietly fail or complain in its own special way
### GIF export may still work, but MP4 absolutely will not.

## Solution:
- Install FFmpeg, you can download the full build from [**here**](https://www.gyan.dev/ffmpeg/builds/)
- Once you have the full build extracted, pop it somewhere permanent and copy the path to the bin file (containing "ffmpeg", "ffplay" and "ffprobe") into your PATH (via Win + R, sysdm.cpl → Advanced → Environment Variables, you can pop it in user or system, up to you).
- Will look something like this in your list of PATH variables.
  <img width="387" height="26" alt="image" src="https://github.com/user-attachments/assets/adc5a21e-3177-409c-8ddf-ed77502b532d" />

- Make sure "ffmpeg" is accessible from the command line by just typing ffmpeg. If it doesn't lose its mind then it's in your PATH and working.
---

## Controls Overview

<img width="1555" height="933" alt="image" src="https://github.com/user-attachments/assets/e31b154e-b7b3-43f1-8a91-787f41c8fd98" />


### View
<img width="662" height="188" alt="image" src="https://github.com/user-attachments/assets/86ab6d5e-ff98-4b2e-b9a5-a7cbe6060b1e" />

- Show grid
- Show axis
- Pick background colour

### Model + Output
<img width="673" height="286" alt="image" src="https://github.com/user-attachments/assets/40279eb5-7994-450e-9055-29c211118b0e" />

- Load OBJ
- Choose MP4 output path
- Also export GIF
- Render clean (disable edges for export)

### Camera (Preview + Render)
<img width="675" height="327" alt="image" src="https://github.com/user-attachments/assets/8aa5ac68-f470-4fd9-a07d-e2d6070e7c9c" />

- Azimuth / Elevation / Distance
- Center (x,y,z)
- Fit camera to model

### Materials / Colours
<img width="675" height="177" alt="image" src="https://github.com/user-attachments/assets/af7c47dd-5c1e-4ce3-9f8f-5cd9994022b8" />

- Pick model face colour
- Pick light colour (tint)

### Lighting
<img width="673" height="395" alt="image" src="https://github.com/user-attachments/assets/9a57bce6-8e96-49fc-a99d-97db4efc6e27" />

- Mode: Headlight or Fixed
- Light azimuth / elevation / distance (fixed mode)
- Copy camera → light
- Brightness + Diffuse sliders

### Display / Shading
<img width="677" height="272" alt="image" src="https://github.com/user-attachments/assets/66308b6a-3ef2-4b3e-8e35-df0cf671a13b" />

- Show faces
- Show edges
- Wireframe-only toggle
- Smooth shading toggle
- Studio preset toggle
- Edge alpha slider

### Turntable Render
<img width="682" height="377" alt="image" src="https://github.com/user-attachments/assets/1b05796d-b4a0-413b-88f0-e594e439a9db" />

- Rotate axis
- Total degrees
- Seconds
- FPS
- Preview 1 frame
- Render animation

---

## Compatibility Notes (aka “Why This Code Looks Like That”)

- Avoids GLMeshItem.meshData() because some pyqtgraph versions don’t expose it reliably.
- Prefers shaders: color or vertexColor for baked lighting, falls back to shaded if needed.
- Render output is a framebuffer grab, which is intentional:
  - what you see is what you export
  - no second renderer to maintain
  - no viewport vs export mismatch horrors

---

## Known Gotchas

- Some OBJs load as scenes with multiple geometries. This tool concatenates what it can via trimesh.
- If your OBJ is huge, it gets normalised to a sane size for camera + turntable.
- If FFmpeg isn’t installed correctly, MP4 export will sulk.

---

## Roadmap

May be exended at some point, obvious next sins include:
- Drag/drop OBJ loading
- Remember last used folders + settings
- Export PNG sequence
- Basic material / texture support (if I want to get into self harm)

---

## License

MIT.  
Let it roam free.  
Dont blame me when it crashes!
