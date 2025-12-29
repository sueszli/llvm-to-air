Breaking down Metal's intermediate representation format
========================================================

[![SamoZ256](https://miro.medium.com/v2/resize:fill:64:64/1*TQl5EHhm9PMo69jeXgkORQ.png)](https://medium.com/@samuliak?source=post_page---byline--41827022489c---------------------------------------)

[SamoZ256](https://medium.com/@samuliak?source=post_page---byline--41827022489c---------------------------------------)

5 min read

·

Dec 27, 2023

[nameless link](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fp%2F41827022489c&operation=register&redirect=https%3A%2F%2Fmedium.com%2F%40samuliak%2Fbreaking-down-metals-intermediate-representation-format-41827022489c&user=SamoZ256&userId=632b41d42268&source=---header_actions--41827022489c---------------------clap_footer------------------)

--

1

[nameless link](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fbookmark%2Fp%2F41827022489c&operation=register&redirect=https%3A%2F%2Fmedium.com%2F%40samuliak%2Fbreaking-down-metals-intermediate-representation-format-41827022489c&source=---header_actions--41827022489c---------------------bookmark_footer------------------)

Listen

Share

![https://llvm.org/Logo.html](https://miro.medium.com/v2/resize:fit:800/format:webp/1*trWdjOhZtcP_-jK2dx1f3g.png)

If you have worked with the [Apple's Metal API](https://developer.apple.com/metal/) before, you know that it uses a custom shading language called [Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf). When working in Xcode, you can have all your shaders in one _Shaders.metal_ file which is precompiled for you by Xcode. But once your application starts growing, you will quickly find the need to divide your shaders into multiple files and precompile them manually.

Let's take a look at simple 2D shader which displays a texture:

The shader is pretty simple, nothing unusual. We can now compile it to the **intermediate format** using terminal: `xcrun -sdk macosx metal -c shaders.metal -o shaders.ir`. This creates a _shader.ir_ file, whose content is not human readable. However, Apple uses a modified version of [clang](https://clang.llvm.org) (C++ compiler) to compile Metal shaders, and clang uses [LLVM](https://llvm.org) for platform agnistic intermediate representation.

This is great, because when can now use an LLVM disassembler to get a human readable version. It should be already install on your computer. So let's disassemble our shader: `llvm-dis shaders.ir -o shaders.ll`.

The file should look something like this:

Phew! That is a lot of code. There is no public documentation of this format (by the time of writing), but most of it is pretty understandable after some reading and experiments. The code is divided into 3 parts, which we are going to look at now.

Header
------

The header are the first 4 lines of the file:

The **source_filename** is pretty self explanatory. I don't fully understand what **target datalayout** is, but it seems like bit sizes and alignment for all the scalar types (e.g. probably something like **i16** has size of **16** and alignment of **16**, while **i1** has size of **8** and alignment of **8**).

**target triple** is a unique identifier for every platform. Interestingly, this target triple (_“air64-apple-macosx14.0.0”_) isn't officially registered in LLVM, so if you would like to use some LLVM tools on this file (for instance LLVM optimizer), you would get an error.

Body
----

The second part of the file is the body. This one is a bit more interesting (and also longer). It contains all the **functions**, **structures** and **code**.

Here we can find our 2 functions (**vertexMain** and **fragmentMain**) as well as 3 other forward declared functions. The code inside the vertex function is pretty simple and doesn't contain anything Metal specific. However, the fragment function has something more to it. First are the arguments. We can notice that there are 2 arguments with an **addrspace** attribute. This attribute corresponds to the Metal address space attribute. **Address space 1** is the equivalent of **device** address space and **address space 2** is **constant**.

Inside the code of the fragment function, there is a call to several functions with the **air** prefix. These are the Metal standard library functions. They are called using the **tail call** instruction, which is an optimization that prevents the called function from creating its own stack memory. Interestingly, Metal uses its own custom functions to perform casting from one type to other instead of using the native LLVM instructions (for instance, the _air.convert.f.v2f32.f.v2f16_ function converts **half2** to **float2**).

The _air.sample_texture_2d.v4f32_ function has 9 parameter:

*   **ptr** — **texture**
*   **ptr** — **sampler**
*   **<2 x float>** — **texture coordinates**
*   **i1** — I have no idea…
*   **<2 x i32>** — **offset**
*   **i1** — once again no idea
*   **float** — probably **bias**
*   **float** — probably **min lod clamp**
*   **i32** — no idea

After the fragment function, we can find forward definitions to the standard library functions and attributes. I am not going to cover the attributes, since there is just too many of them and I am also not sure what some of them do.

Metadata
--------

This is the last section of the file and is arguably the most interesting one. This section describes how should the body be treated. It includes things like entry point names, binding locations of buffers, textures and samplers, but also debug information, for instance the name of the compiler and language.

The part which is used by Metal is the following:

This sections then references data in the later section. I will go line by line.

### LLVM module flags

This section contains some general information about the code, mostly the device limits. I also don't really know what does the **frame-pointer** line mean.

### AIR vertex and AIR fragment

This is probably the longest and most complicated section. The entry point definition (_!9_ and _!16_) has 3 arguments:

*   **reference to the function**
*   **outputs**
*   **inputs**

The outputs and inputs have always at least these 4 arguments:

*   **air.arg_type_name** — specifies that the next argument will be the name of the type
*   **type name** — the name of the type (for debugging)
*   **air.arg_name** — specifies that the next argument will be the name of the argument, not present when it is output and not a structure (see _!18_ in the code)
*   **name** — the name of the argument (for debugging)

Additionally, every input begins with an **i32 index**, which is most likely just the index of the argument as it appears in the function. There is no index in case of output, so I assume that it is matched based on the order that they appear in the outputs array (_!10_ and _!17_).

Every input and output has its type (air.position, air.vertex_output, air.texture, etc), which is an information about how it is used. Based on this, there can be some additional arguments. We are going to look at them individually:

*   **position** (in case of vertex shader) — no additional arguments
*   **vertex_output** — has 1 argument (whose value can for example be **generated(8texCoordDv2_Dh)**), but I don't know what is its purpose
*   **vertex_input** — has 3 arguments: **“air.location_index”**, the location index (specified with _[[attribute(X)]]_ in Metal) and an **i32** whose purpose I don't know
*   render_target — has 2 arguments: the render target index (optionally specified by _[[color(X)]]_ in Metal) and an **i32**
*   **position** (in case of fragment shader) — has 2 arguments: 1. probably specifies where are the _(0, 0)_ coordinates 2. most likely has something to do with perspective division
*   **fragment_input** — same as **vertex_output**
*   **texture** — has 3 arguments: “air.location_index”, the location index (specified with [[texture(X)]] in Metal) and an **i32** + template arguments (in this case we can see **“air.sample”**, because we specified the **access::sample** parameter in Metal)
*   **sampler** — has 3 arguments: “air.location_index”, the location index (specified with [[sampler(X)]] in Metal) and an **i32** + template arguments

There are a lot more types, we have barely scratched the surface. But you should now have some idea about what all of that magic stuff means.

### AIR compile options

These are the options that were used when the code was compiled. It is for debugging purposes.

### LLVM ident

This is the identifier of the compiler. Not much to say here.

### AIR version

This is the version of the AIR format used (major, minor, patch).

### AIR language version

This is the name and version of the source language.

### AIR source file name

This is the filename of the source file.

### What about the rest?

Regarding the last 4 lines of metadata, I still haven't figured out what is their purpose. They seem to be pretty random and they aren't referenced by anything.

Closing thoughts
----------------

Phew! That was a lot of work! Reading this code isn't easy, and the fact that its undocumented doesn't help it. I hope you found this article useful and be sure to check out my shading language compiler [here](https://github.com/SamoZ256/lvslang). Cheers!

_This article is free to read. If you found it engaging, please consider supporting my work through_ [_GitHub Sponsors_](https://github.com/sponsors/SamoZ256)_._