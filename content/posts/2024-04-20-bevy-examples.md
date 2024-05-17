+++
title = "Bevy Examples"
date = 2024-04-20
description = ""

[taxonomies]
tags = ["Bevy", "Game"]
+++


简单分析下这些栗子，为后续的小游戏做准备：
https://github.com/bevyengine/bevy/blob/main/examples/2d/2d_shapes.rs#L8

# 2d_shapes.rs
简单是Setup一下，把相机📷设定到2D，添加不同形状的2D mesh以及纯色Material

# 2d_viewport_to_world.rs
画一个跟随鼠标的Gizmos

# bloom_2d.rs
做了一个可以通过键盘调节泛光效果参数的栗子

### ChatGPT 解释
Bloom effect（泛光效果）是一种在数字图像和视频游戏中常见的视觉效果，用来模拟在现实生活中由于相机镜头或人眼对光线处理方式而产生的一种现象。当在照片或视频中遇到非常亮的光源时，这些光源周围会出现一种光晕或发光的效果，这就是所谓的泛光效果。

### 泛光效果的工作原理
泛光效果通常通过以下步骤实现：
1. **提取高亮**：首先从图像中提取出高亮区域，通常是那些亮度超过一定阈值的部分。
2. **模糊处理**：将提取出的高亮区域进行模糊处理，使得光源周围形成光晕。
3. **叠加处理**：将模糊后的图像与原始图像叠加，创建出光源周围的发光效果。

### 应用场景
泛光效果在多种应用场景中都非常普遍，例如：
- **电影和电视**：为了增强画面的视觉冲击力或表达特定的氛围和情感。
- **视频游戏**：增加游戏的视觉真实感和美感，尤其是在表现强烈光源或魔法效果时。
- **图形设计**：在广告或艺术作品中创建特定的视觉风格。

### 技术实现
在计算机图形学中，泛光效果通常通过着色器（Shaders）实现。着色器是运行在图形处理器（GPU）上的程序，能够高效地处理图像数据，生成泛光等效果。在游戏引擎中，如Unity或Unreal Engine，通常提供了内置的泛光效果工具，允许开发者轻松地添加和配置泛光效果，以适应不同的游戏场景和光照条件。

总之，泛光效果是数字图像处理中一个重要且常用的技术，它通过模拟现实世界中的光线散射现象，增强了视觉内容的丰富性和吸引力。


# bounding_2d.rs
展示了不同的拦截检测的方法。

有个值得注意的点是生成宏`Default`可以标记`Enum`的默认选项

```rust
#[derive(States, Default, Debug, Hash, PartialEq, Eq, Clone, Copy)]
enum Test {
    AabbSweep,
    CircleSweep,
    #[default]
    RayCast,
    AabbCast,
    CircleCast,
}
```

还有个比较有意思的点是ECS的Behavior的`.run_if`

```rust
ray_cast_system.run_if(in_state(Test::RayCast))
```

### Gizmo vs Gizmos

#### Gizmo 辅助工具
在计算机图形和游戏开发中，"Gizmo"（或称为控制小工具）是一种可视化工具，用于在编辑环境中帮助开发者以交互方式操控和调整场景对象的属性。Gizmo 通常用于表示和编辑物体的位置、旋转、缩放等变换属性。

##### Gizmo 的主要用途：

1. **变换控制**：
   - Gizmo 允许用户通过直观的图形界面（通常是三维箭头和旋转圈）来移动、旋转和缩放场景中的对象。
   - 这些控件映射到对象的本地或世界坐标系，使编辑更加直观和精确。

2. **视觉调试**：
   - 在开发过程中，Gizmo 用于显示诸如力的方向、速度向量、碰撞体积等不易直观理解的抽象概念。
   - 这可以帮助开发者理解游戏或应用的行为，并对其进行调整。

3. **编辑器功能**：
   - 在游戏引擎和3D建模软件中，Gizmo 是标准工具，用于在设计阶段调整和优化模型和场景布局。
   - Gizmo 提高了工作流程的效率，使得复杂场景的编辑变得更加容易。

##### Gizmo 的类型：
- **位置Gizmo**：允许用户通过拖动箭头沿特定轴移动对象。
- **旋转Gizmo**：展示环形控件，用户可以围绕一轴旋转对象。
- **缩放Gizmo**：提供可拖动的手柄，用于在一个或多个方向上调整对象的大小。
- **自定义Gizmo**：针对特定应用或游戏开发任务定制的工具，如路径编辑、光线投射等。

在开发环境中，有效使用Gizmo 是提高生产效率、确保精确配置和增强场景交互性的关键工具。

#### Gizmos 辅助线
在编程和游戏开发中，**Gizmos** 是用于在编辑模式或调试时在3D场景中可视化各种辅助线和几何形状的工具。它们帮助开发者直观地看到非直观的数据，如碰撞区域、力的方向、光线投射或者对象的轴。Gizmos 不会在最终的游戏或应用程序中呈现，它们仅在开发过程中用于辅助设计和调试。

##### Gizmos的主要功能包括：

1. **视觉辅助**：
   - Gizmos通过在编辑器视图中绘制简单的图形（如线条、圆圈、盒子等）来显示关于对象如何与其他对象或环境互动的信息。这对于调整物理引擎参数、摄像机视角、光照位置等非常有用。

2. **交互控制**：
   - 在3D建模和游戏开发软件中，Gizmos提供了一种操作对象的直观方式。例如，使用平移、旋转和缩放Gizmos，开发者可以精确地修改对象的位置、姿态和大小。

3. **调试和优化**：
   - Gizmos是调试过程中不可或缺的工具，可以用来显示不可见的游戏逻辑，如导航网格、感知觉野或者动态生成的路径。这有助于开发者理解和解决潜在的问题。

### 示例用途：
- 在Unity或Unreal Engine这类游戏引擎中，开发者可以通过编程方式添加Gizmos，以在游戏世界中标示特定的位置或显示对象的边界框。
- 在物理模拟中，可以使用Gizmos来可视化力的应用点和方向。
- 在路径规划或AI开发中，Gizmos可以用来表示AI的视觉或听觉范围。

总的来说，Gizmos是游戏开发和3D建模中一个非常重要的工具，它们通过提供直观的视觉参考来辅助开发和调试过程，虽然在最终产品中通常是看不见的。

# custom_gltf_vertex_attribute.rs

可以注意一下这里会返回一个`Builder`，后续可以继续set其他的`Plugin`

```
DefaultPlugins.set
```

另外在这个example里加载了gltf和wgsl两个资源，这里`mix`的第三个参数是混合比例，

而在`smoothstep`中控制了当前点的重心坐标最小值和时间的关系，通过时间来对图形进行裁剪。

`t>d`的部分为`0`，显示白色，`t+0.01<d`的部分为默认材质。如果调整`t+0.01`为`t+0.5`可以看到边缘合并的效果。

```wgsl
@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    let d = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
    let t = 0.25 * (0.85 + sin(1.0 * globals.time));
    return mix(vec4(1.0,0.0,1.0,1.0), input.color, smoothstep(t, t+0.01, d));
}
```

# mesh2d.rs
最基本的setup，画了个紫色方块

# mesh2d_manual.rs
画了一个黄色的五角星

## 通过手动构建顶点的方式造了一个五角星

## 写了一个Plugin
在plugin中注册了`ExtractSchedule` `Render`两个阶段，分别手动实现了对象的提取和加载渲染对象到渲染队列的过程

# mesh2d_vertex_color_texture.rs
正方形上画顶点色或加载图片

# move_sprite.rs
加载一张图片并形成一个`SpritBundle`，如何在`Update`里移动其位置

# pixel_grid_snap.rs
关闭MSAA之后旋转Sprit

# rotation.rs
一个极简化版本的塔防，四个位置的炮台会始终尝试瞄准移动物

# sprite_sheet.rs
加载PNG，使用 `TextureAtlasLayout`+`TextureAtlas`， 在Update的时候切换部分PNG内容来进行动画的播放

# sprite_slice.rs
对材质做各种变换和处理的Demo

# sprite_tile.rs
材质平铺Demo

# text2d.rs
文字的各种变换

# texture_atlas.rs
