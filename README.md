# rwx2blender

Blender add-on to import Active Worlds RenderWare scripts (.rwx)

## What's RenderWare?

RenderWare (RW), developed by Criterion Software Limited, is the name of the 3D API graphics rendering engine used in the on-line, 3D, virtual reality and interactive environment Active Worlds (AW)

## What's a RenderWare script?

A RenderWare script (RWX) file is an ASCII text file of an object/model's geometry.
This geometry consists of vertices, polygons, and material information. 

### Installing

First of all: clone this project, open a terminal and type:
```
git clone https://github.com/Blaxar/rwx2blender.git
```

Next, to import the add-on in Blender: go to *File* > *User Preferences...* then clic the *Add-ons* tab and clic *Install add-on frome file...*, select *rwx2blender.py* from the freshly cloned repository and import it.

Finally, to enable the plugin once it's imported, check the box in its entry in the *Add-ons* tab, if you have trouble finding it, select the *User* category to filter the list.

### Usage

To import a RenderWare script (.rwx) in Blender, go to *File* > *Import* > *Active Worlds RenderWare script (.rwx)*

If you don't specify a texture folder (using the entry on the left in the pop-up window), it will assume one for you based in the usual Active Worlds path folder hierarchy, for instance, let's say you want to import this file:

```
/custom/path/rwx/furniture.rwx
```

The assumed path for the texture folder will be:

```
/custom/path/textures
```

### References:

http://www.tnlc.com/rw/rwx.html