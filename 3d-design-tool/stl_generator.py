"""
STL Generator - 複合形状・複雑造形対応版
テキストパラメータから複雑な3Dメッシュを生成してSTLとして出力する
Bambu Lab mini 対応 (標準バイナリSTL形式)
"""

import trimesh
import trimesh.boolean
import numpy as np

try:
    from shapely import geometry as shapely_geometry
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def generate_stl(shape_params: dict) -> bytes:
    """
    形状パラメータからSTLバイナリデータを生成する

    Args:
        shape_params: LLMから返される形状パラメータ
            単一形状: {"shape": "...", "dimensions": {...}}
            複合形状: {"shape": "compound", "components": [...]}

    Returns:
        バイナリSTLデータ
    """
    shape = shape_params.get("shape", "box").lower()
    dims = shape_params.get("dimensions", {})
    components = shape_params.get("components", [])

    if shape == "compound" and components:
        mesh = _create_compound(components)
    else:
        mesh = _create_shape(shape, dims)

    # メッシュを正規化
    try:
        mesh.process()
    except Exception:
        pass

    # Bambu Lab用: メッシュの底面をZ=0に配置
    bounds = mesh.bounds
    if bounds is not None and np.asarray(bounds).ndim == 2:
        z_min = float(bounds[0][2])
        if z_min < 0:
            mesh.apply_translation([0, 0, -z_min])

    return mesh.export(file_type="stl")


# ================================================================
# 複合形状 (Compound)
# ================================================================

def _create_compound(components: list) -> trimesh.Trimesh:
    """
    複数のプリミティブをブーリアン演算で合成する

    各コンポーネント:
      {
        "shape": "cylinder",
        "dimensions": {"radius": 5, "height": 20},
        "position": [x, y, z],   # コンポーネント中心の座標 (mm)
        "rotation": [rx, ry, rz], # 回転角度 (度)
        "operation": "union" | "subtract" | "intersect"
      }
    """
    result = None

    for comp in components:
        shape = comp.get("shape", "box").lower()
        dims = comp.get("dimensions", {})
        pos = [float(v) for v in comp.get("position", [0, 0, 0])]
        rot_deg = [float(v) for v in comp.get("rotation", [0, 0, 0])]
        operation = comp.get("operation", "union").lower()

        try:
            mesh = _create_shape(shape, dims)
        except Exception:
            continue

        # 回転を適用
        if any(r != 0 for r in rot_deg):
            rx, ry, rz = [np.radians(r) for r in rot_deg]
            T = trimesh.transformations.euler_matrix(rx, ry, rz, "rxyz")
            mesh.apply_transform(T)

        # 平行移動を適用
        if any(p != 0 for p in pos):
            mesh.apply_translation(pos)

        if result is None:
            result = mesh
            continue

        if operation == "subtract":
            try:
                new = trimesh.boolean.difference([result, mesh])
                if new is not None and len(new.vertices) > 0:
                    result = new
            except Exception:
                pass  # subtract 失敗時はスキップ

        elif operation == "intersect":
            try:
                new = trimesh.boolean.intersection([result, mesh])
                if new is not None and len(new.vertices) > 0:
                    result = new
            except Exception:
                pass

        else:  # union
            try:
                new = trimesh.boolean.union([result, mesh])
                if new is not None and len(new.vertices) > 0:
                    result = new
                else:
                    raise ValueError("empty union")
            except Exception:
                # boolean が使えない場合は単純結合 (フォールバック)
                result = trimesh.util.concatenate([result, mesh])

    return result if result is not None else trimesh.creation.box(extents=[20, 20, 20])


# ================================================================
# 形状ディスパッチ
# ================================================================

def _create_shape(shape: str, dims: dict) -> trimesh.Trimesh:
    """形状タイプに応じてtrimeshメッシュを生成する"""

    # --- 直方体 / 立方体 ---
    if shape in ["box", "cube", "直方体", "立方体", "rectangular", "rectangle"]:
        width = float(dims.get("width", dims.get("w", 20)))
        depth = float(dims.get("depth", dims.get("d", 20)))
        height = float(dims.get("height", dims.get("h", 20)))
        return trimesh.creation.box(extents=[width, depth, height])

    # --- 角丸直方体 ---
    elif shape in ["rounded_box", "rounded_cube", "丸角直方体", "角丸ボックス"]:
        width = float(dims.get("width", dims.get("w", 20)))
        depth = float(dims.get("depth", dims.get("d", 20)))
        height = float(dims.get("height", dims.get("h", 20)))
        radius = float(dims.get("radius", dims.get("r", min(width, depth, height) * 0.1)))
        return _create_rounded_box(width, depth, height, radius)

    # --- 球体 ---
    elif shape in ["sphere", "球", "球体", "ball"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        subdivisions = int(dims.get("subdivisions", 4))
        return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

    # --- 半球 ---
    elif shape in ["hemisphere", "半球", "dome", "ドーム"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        subdivisions = int(dims.get("subdivisions", 4))
        return _create_hemisphere(radius, subdivisions)

    # --- 楕円体 ---
    elif shape in ["ellipsoid", "楕円体", "oval", "egg"]:
        rx = float(dims.get("radius_x", dims.get("rx", 15)))
        ry = float(dims.get("radius_y", dims.get("ry", 10)))
        rz = float(dims.get("radius_z", dims.get("rz", 8)))
        subdivisions = int(dims.get("subdivisions", 4))
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        sphere.vertices *= np.array([rx, ry, rz])
        return sphere

    # --- 円柱 ---
    elif shape in ["cylinder", "円柱", "シリンダー", "tube"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        height = float(dims.get("height", dims.get("h", 20)))
        sections = int(dims.get("sections", 64))
        return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    # --- 円錐 ---
    elif shape in ["cone", "円錐", "コーン"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        height = float(dims.get("height", dims.get("h", 20)))
        sections = int(dims.get("sections", 64))
        return trimesh.creation.cone(radius=radius, height=height, sections=sections)

    # --- 台形円錐 (フラストム) ---
    elif shape in ["frustum", "truncated_cone", "台形円錐", "フラストム"]:
        bottom_radius = float(dims.get("bottom_radius", dims.get("radius", 15)))
        top_radius = float(dims.get("top_radius", bottom_radius * 0.5))
        height = float(dims.get("height", dims.get("h", 20)))
        sections = int(dims.get("sections", 64))
        return _create_frustum(bottom_radius, top_radius, height, sections)

    # --- トーラス / ドーナツ ---
    elif shape in ["torus", "トーラス", "ドーナツ", "donut", "ring"]:
        major_radius = float(
            dims.get("major_radius", dims.get("outer_radius", dims.get("R", 15)))
        )
        minor_radius = float(
            dims.get("minor_radius", dims.get("tube_radius", dims.get("r", 4)))
        )
        major_sections = int(dims.get("major_sections", 64))
        minor_sections = int(dims.get("minor_sections", 32))
        return trimesh.creation.torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_sections=major_sections,
            minor_sections=minor_sections,
        )

    # --- カプセル ---
    elif shape in ["capsule", "カプセル", "pill"]:
        radius = float(dims.get("radius", dims.get("r", 8)))
        height = float(dims.get("height", dims.get("h", 30)))
        count = int(dims.get("count", 32))
        return trimesh.creation.capsule(radius=radius, height=height, count=[count, count])

    # --- ピラミッド ---
    elif shape in ["pyramid", "ピラミッド"]:
        base = float(dims.get("base", dims.get("width", 20)))
        height = float(dims.get("height", dims.get("h", 20)))
        return _create_pyramid(base, height)

    # --- 星形 ---
    elif shape in ["star", "星", "star_prism", "星形"]:
        n_points = int(dims.get("points", dims.get("n_points", 5)))
        outer_radius = float(dims.get("outer_radius", dims.get("radius", dims.get("r", 15))))
        inner_radius = float(dims.get("inner_radius", outer_radius * 0.4))
        height = float(dims.get("height", dims.get("h", 10)))
        return _create_star_prism(n_points, outer_radius, inner_radius, height)

    # --- 矢印 ---
    elif shape in ["arrow", "矢印", "アロー"]:
        total_length = float(dims.get("total_length", dims.get("length", 40)))
        shaft_width = float(dims.get("shaft_width", total_length * 0.2))
        head_width = float(dims.get("head_width", total_length * 0.4))
        head_length = float(dims.get("head_length", total_length * 0.35))
        thickness = float(dims.get("thickness", dims.get("height", 8)))
        return _create_arrow(total_length, shaft_width, head_width, head_length, thickness)

    # --- 十字形 ---
    elif shape in ["cross", "十字", "plus", "クロス", "十字形"]:
        arm_length = float(dims.get("arm_length", dims.get("length", 30)))
        arm_width = float(dims.get("arm_width", dims.get("width", 10)))
        height = float(dims.get("height", dims.get("h", 8)))
        return _create_cross(arm_length, arm_width, height)

    # --- 歯車 ---
    elif shape in ["gear", "歯車", "ギア"]:
        n_teeth = int(dims.get("teeth", dims.get("n_teeth", 12)))
        module = float(dims.get("module", dims.get("m", 2.0)))
        thickness = float(dims.get("thickness", dims.get("height", 10)))
        bore_radius = float(dims.get("bore_radius", dims.get("hole_radius", 0)))
        return _create_gear(n_teeth, module, thickness, bore_radius)

    # --- スプリング / コイル ---
    elif shape in ["spring", "スプリング", "コイル", "coil", "helix"]:
        coil_radius = float(dims.get("coil_radius", dims.get("radius", 10)))
        wire_radius = float(dims.get("wire_radius", dims.get("wire_r", 1.5)))
        num_coils = float(dims.get("num_coils", dims.get("coils", 5)))
        pitch = float(dims.get("pitch", wire_radius * 3.5))
        return _create_spring(coil_radius, wire_radius, num_coils, pitch)

    # --- L字ブラケット ---
    elif shape in ["l_bracket", "l_shape", "l字", "angle_bracket", "Lブラケット"]:
        arm1 = float(dims.get("arm1_length", dims.get("height", 30)))
        arm2 = float(dims.get("arm2_length", dims.get("width", 30)))
        thickness = float(dims.get("thickness", 3))
        depth = float(dims.get("depth", dims.get("d", 20)))
        return _create_l_bracket(arm1, arm2, thickness, depth)

    # --- T字ブラケット ---
    elif shape in ["t_bracket", "t_shape", "t字", "Tブラケット"]:
        top_length = float(dims.get("top_length", dims.get("width", 40)))
        stem_length = float(dims.get("stem_length", dims.get("height", 30)))
        arm_width = float(dims.get("arm_width", dims.get("thickness", 8)))
        depth = float(dims.get("depth", dims.get("d", 10)))
        return _create_t_bracket(top_length, stem_length, arm_width, depth)

    # --- 扇形柱 (ウェッジ) ---
    elif shape in ["wedge", "扇形", "sector", "arc_prism", "ウェッジ"]:
        radius = float(dims.get("radius", dims.get("r", 15)))
        angle = float(dims.get("angle", 90))
        height = float(dims.get("height", dims.get("h", 10)))
        return _create_wedge(radius, angle, height)

    # --- 中空円柱 (パイプ/チューブ) ---
    elif shape in ["pipe", "hollow_cylinder", "パイプ", "中空円柱"]:
        outer_radius = float(dims.get("outer_radius", dims.get("radius", 12)))
        inner_radius = float(dims.get("inner_radius", dims.get("inner_r", 8)))
        height = float(dims.get("height", dims.get("h", 30)))
        sections = int(dims.get("sections", 64))
        return _create_hollow_cylinder(outer_radius, inner_radius, height, sections)

    # --- 六角柱 ---
    elif shape in ["hexagonal_prism", "六角柱", "hex_prism", "hexagon"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        height = float(dims.get("height", dims.get("h", 20)))
        return trimesh.creation.cylinder(radius=radius, height=height, sections=6)

    # --- 花瓶 ---
    elif shape in ["vase", "花瓶", "ボトル", "bottle", "jar"]:
        height = float(dims.get("height", dims.get("h", 120)))
        bottom_radius = float(dims.get("bottom_radius", dims.get("base_radius", 25)))
        max_radius = float(dims.get("max_radius", dims.get("belly_radius", bottom_radius * 1.6)))
        top_radius = float(dims.get("top_radius", dims.get("mouth_radius", bottom_radius * 0.8)))
        wall = float(dims.get("wall_thickness", dims.get("wall", 3)))
        return _create_vase(height, bottom_radius, max_radius, top_radius, wall)

    # --- ボウル / 皿 ---
    elif shape in ["bowl", "ボウル", "皿", "dish", "cup_open"]:
        outer_radius = float(dims.get("outer_radius", dims.get("radius", dims.get("r", 50))))
        depth = float(dims.get("depth", dims.get("height", outer_radius * 0.6)))
        wall = float(dims.get("wall_thickness", dims.get("wall", 3)))
        return _create_bowl(outer_radius, depth, wall)

    # --- ねじれ柱 ---
    elif shape in ["twisted_prism", "twisted_column", "ねじれ柱", "螺旋柱", "helix_column"]:
        sides = int(dims.get("sides", dims.get("n_sides", 6)))
        radius = float(dims.get("radius", dims.get("r", 15)))
        height = float(dims.get("height", dims.get("h", 80)))
        twist = float(dims.get("twist_angle", dims.get("twist", 120)))
        return _create_twisted_prism(sides, radius, height, twist)

    # --- 波面プレート ---
    elif shape in ["wavy_plate", "wave_surface", "波プレート", "波板", "sine_plate"]:
        width = float(dims.get("width", dims.get("w", 80)))
        depth = float(dims.get("depth", dims.get("d", 80)))
        base_h = float(dims.get("base_height", dims.get("thickness", 5)))
        amplitude = float(dims.get("amplitude", base_h * 1.2))
        wave_count = float(dims.get("wave_count", dims.get("waves", 3)))
        return _create_wavy_plate(width, depth, base_h, amplitude, wave_count)

    # --- デフォルト: 直方体 ---
    else:
        return trimesh.creation.box(extents=[20, 20, 20])


# ================================================================
# 各形状の実装
# ================================================================

def _create_rounded_box(width: float, depth: float, height: float, radius: float) -> trimesh.Trimesh:
    """角丸直方体"""
    try:
        r = min(float(radius), min(width, depth, height) / 2 - 0.01)
        r = max(r, 0.1)
        return trimesh.creation.roundedbox(extents=[width, depth, height], radius=r)
    except Exception:
        return trimesh.creation.box(extents=[width, depth, height])


def _create_hemisphere(radius: float, subdivisions: int = 4) -> trimesh.Trimesh:
    """半球"""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    try:
        result = trimesh.intersections.slice_mesh_plane(
            sphere,
            plane_normal=[0, 0, -1],
            plane_origin=[0, 0, 0],
            cap=True,
        )
        if result is not None and len(result.vertices) > 0:
            return result
    except Exception:
        pass
    # フォールバック: z >= 0 の面だけ保持
    vertices = sphere.vertices
    faces = sphere.faces
    upper = vertices[:, 2] >= -0.01
    face_mask = upper[faces].all(axis=1)
    if face_mask.any():
        return trimesh.Trimesh(vertices=vertices, faces=faces[face_mask])
    return sphere


def _create_frustum(bottom_radius: float, top_radius: float, height: float, sections: int = 64) -> trimesh.Trimesh:
    """台形円錐 (フラストム / 切頭円錐)"""
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)

    bottom_verts = np.column_stack([
        bottom_radius * np.cos(theta),
        bottom_radius * np.sin(theta),
        np.zeros(sections),
    ])
    top_verts = np.column_stack([
        top_radius * np.cos(theta),
        top_radius * np.sin(theta),
        np.full(sections, height),
    ])

    bottom_center = np.array([[0.0, 0.0, 0.0]])
    top_center = np.array([[0.0, 0.0, float(height)]])
    vertices = np.vstack([bottom_center, bottom_verts, top_verts, top_center])

    faces = []
    bc = 0
    tc = 2 * sections + 1
    for i in range(sections):
        ni = (i + 1) % sections
        b0, b1 = i + 1, ni + 1
        t0, t1 = sections + i + 1, sections + ni + 1
        faces.append([bc, b1, b0])    # bottom cap
        faces.append([b0, b1, t0])    # side
        faces.append([b1, t1, t0])    # side
        faces.append([tc, t0, t1])    # top cap

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int32))
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _create_star_prism(n_points: int, outer_radius: float, inner_radius: float, height: float) -> trimesh.Trimesh:
    """星形プリズム"""
    if not SHAPELY_AVAILABLE:
        return trimesh.creation.cylinder(radius=outer_radius, height=height, sections=n_points * 2)

    n = n_points * 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    radii = [outer_radius if i % 2 == 0 else inner_radius for i in range(n)]
    coords = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
    polygon = shapely_geometry.Polygon(coords)
    return trimesh.creation.extrude_polygon(polygon, height)


def _create_arrow(
    total_length: float,
    shaft_width: float,
    head_width: float,
    head_length: float,
    thickness: float,
) -> trimesh.Trimesh:
    """矢印形状 (Z軸方向に押し出し)"""
    if not SHAPELY_AVAILABLE:
        return trimesh.creation.cone(radius=head_width / 2, height=total_length)

    shaft_length = total_length - head_length
    hw = shaft_width / 2
    hew = head_width / 2
    coords = [
        (-hw, 0),
        (-hw, shaft_length),
        (-hew, shaft_length),
        (0, total_length),
        (hew, shaft_length),
        (hw, shaft_length),
        (hw, 0),
    ]
    polygon = shapely_geometry.Polygon(coords)
    return trimesh.creation.extrude_polygon(polygon, thickness)


def _create_cross(arm_length: float, arm_width: float, height: float) -> trimesh.Trimesh:
    """十字形プリズム"""
    if not SHAPELY_AVAILABLE:
        return trimesh.creation.box(extents=[arm_length, arm_width, height])

    half_w = arm_width / 2
    half_l = arm_length / 2
    h_bar = shapely_geometry.Polygon([
        (-half_l, -half_w), (half_l, -half_w),
        (half_l, half_w), (-half_l, half_w),
    ])
    v_bar = shapely_geometry.Polygon([
        (-half_w, -half_l), (half_w, -half_l),
        (half_w, half_l), (-half_w, half_l),
    ])
    cross_poly = h_bar.union(v_bar)
    return trimesh.creation.extrude_polygon(cross_poly, height)


def _create_gear(n_teeth: int, module: float, thickness: float, bore_radius: float = 0) -> trimesh.Trimesh:
    """平歯車 (矩形歯形の近似)"""
    pitch_radius = n_teeth * module / 2
    addendum = module
    dedendum = 1.25 * module
    outer_radius = pitch_radius + addendum
    root_radius = max(pitch_radius - dedendum, module * 0.5)

    if not SHAPELY_AVAILABLE:
        return trimesh.creation.cylinder(radius=outer_radius, height=thickness, sections=n_teeth * 2)

    # 歯先・歯底を交互に配置した多角形
    n = n_teeth * 4
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # 歯先2点・歯底2点の繰り返し
    radii = [outer_radius if i % 4 < 2 else root_radius for i in range(n)]
    coords = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
    gear_poly = shapely_geometry.Polygon(coords)

    if bore_radius > 0:
        bore = shapely_geometry.Point(0, 0).buffer(bore_radius, resolution=32)
        gear_poly = gear_poly.difference(bore)

    return trimesh.creation.extrude_polygon(gear_poly, thickness)


def _create_spring(
    coil_radius: float,
    wire_radius: float,
    num_coils: float,
    pitch: float,
    segments_per_coil: int = 36,
) -> trimesh.Trimesh:
    """コイルスプリング"""
    total_height = num_coils * pitch
    n_points = max(int(num_coils * segments_per_coil), 64)
    t = np.linspace(0, num_coils * 2 * np.pi, n_points)

    path = np.column_stack([
        coil_radius * np.cos(t),
        coil_radius * np.sin(t),
        np.linspace(0, total_height, n_points),
    ])

    if SHAPELY_AVAILABLE:
        circle_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        circle_coords = [
            (wire_radius * np.cos(a), wire_radius * np.sin(a))
            for a in circle_angles
        ]
        circle_poly = shapely_geometry.Polygon(circle_coords)
        try:
            return trimesh.creation.sweep_polygon(circle_poly, path)
        except Exception:
            pass

    # フォールバック: トーラス近似
    return trimesh.creation.torus(major_radius=coil_radius, minor_radius=wire_radius)


def _create_l_bracket(
    arm1_length: float, arm2_length: float, thickness: float, depth: float
) -> trimesh.Trimesh:
    """L字型ブラケット"""
    if not SHAPELY_AVAILABLE:
        return trimesh.creation.box(extents=[arm2_length, depth, arm1_length])

    vert_arm = shapely_geometry.Polygon([
        (0, 0), (thickness, 0), (thickness, arm1_length), (0, arm1_length)
    ])
    horiz_arm = shapely_geometry.Polygon([
        (0, 0), (arm2_length, 0), (arm2_length, thickness), (0, thickness)
    ])
    l_poly = vert_arm.union(horiz_arm)
    return trimesh.creation.extrude_polygon(l_poly, depth)


def _create_t_bracket(
    top_length: float, stem_length: float, arm_width: float, depth: float
) -> trimesh.Trimesh:
    """T字型ブラケット"""
    if not SHAPELY_AVAILABLE:
        return trimesh.creation.box(extents=[top_length, depth, stem_length + arm_width])

    half_top = top_length / 2
    half_stem = arm_width / 2
    # 上部の横棒
    top_bar = shapely_geometry.Polygon([
        (-half_top, 0), (half_top, 0),
        (half_top, arm_width), (-half_top, arm_width),
    ])
    # 下部の縦棒
    stem = shapely_geometry.Polygon([
        (-half_stem, 0), (half_stem, 0),
        (half_stem, arm_width + stem_length), (-half_stem, arm_width + stem_length),
    ])
    t_poly = top_bar.union(stem)
    return trimesh.creation.extrude_polygon(t_poly, depth)


def _create_wedge(radius: float, angle_deg: float, height: float) -> trimesh.Trimesh:
    """扇形柱 (ウェッジ)"""
    if not SHAPELY_AVAILABLE:
        sections = max(3, int(angle_deg / 10))
        return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    angle_rad = np.radians(angle_deg)
    n_points = max(3, int(angle_deg / 3))
    angles = np.linspace(0, angle_rad, n_points)
    coords = [(0.0, 0.0)]
    coords += [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    polygon = shapely_geometry.Polygon(coords)
    return trimesh.creation.extrude_polygon(polygon, height)


def _create_surface_of_revolution(
    r_profile: np.ndarray,
    z_profile: np.ndarray,
    sections: int = 64,
) -> trimesh.Trimesh:
    """プロファイル曲線をZ軸周りに回転させてソリッドを生成する"""
    n = len(r_profile)
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)

    rings = []
    for r, z in zip(r_profile, z_profile):
        ring = np.column_stack([
            r * np.cos(angles),
            r * np.sin(angles),
            np.full(sections, float(z)),
        ])
        rings.append(ring)

    bc = np.array([[0.0, 0.0, float(z_profile[0])]])
    tc = np.array([[0.0, 0.0, float(z_profile[-1])]])
    all_verts = np.vstack(rings + [bc, tc])
    bc_idx = n * sections
    tc_idx = n * sections + 1

    faces = []

    def vi(p, s):
        return p * sections + int(s) % sections

    for p in range(n - 1):
        for s in range(sections):
            v00, v01 = vi(p, s), vi(p, s + 1)
            v10, v11 = vi(p + 1, s), vi(p + 1, s + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    for s in range(sections):
        faces.append([bc_idx, vi(0, s + 1), vi(0, s)])

    for s in range(sections):
        faces.append([tc_idx, vi(n - 1, s), vi(n - 1, s + 1)])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces, dtype=np.int32))
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _vase_profile(t: np.ndarray, r_bottom: float, r_max: float, r_top: float) -> np.ndarray:
    """スムーズな花瓶プロファイル曲線 (smoothstep 補間)"""
    def smoothstep(x):
        return x * x * (3 - 2 * x)

    r = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < 0.35:
            s = smoothstep(ti / 0.35)
            r[i] = r_bottom + (r_max - r_bottom) * s
        elif ti < 0.65:
            neck = r_bottom * 0.55 + r_max * 0.25 + r_top * 0.20
            s = smoothstep((ti - 0.35) / 0.30)
            r[i] = r_max + (neck - r_max) * s
        else:
            neck = r_bottom * 0.55 + r_max * 0.25 + r_top * 0.20
            s = smoothstep((ti - 0.65) / 0.35)
            r[i] = neck + (r_top - neck) * s
    return r


def _create_vase(
    height: float,
    bottom_radius: float,
    max_radius: float,
    top_radius: float,
    wall_thickness: float = 3.0,
    sections: int = 64,
    n_profile: int = 40,
) -> trimesh.Trimesh:
    """
    花瓶 / ボトル形状 (回転体)
    外側と内側の回転体をブーリアン差分で中空化する。
    """
    t = np.linspace(0, 1, n_profile)
    z = height * t

    r_outer = _vase_profile(t, bottom_radius, max_radius, top_radius)
    outer_mesh = _create_surface_of_revolution(r_outer, z, sections)

    # 内側キャビティ: wall_thickness 分だけ細く、bottom_from から開始
    r_inner = np.maximum(r_outer - wall_thickness, wall_thickness * 0.3)
    inner_h = height - wall_thickness + 1.0   # 貫通させるため少し長く
    t_inner = np.linspace(0, 1, n_profile)
    z_inner = inner_h * t_inner
    inner_mesh = _create_surface_of_revolution(r_inner, z_inner, sections)
    inner_mesh.apply_translation([0, 0, wall_thickness])

    try:
        result = trimesh.boolean.difference([outer_mesh, inner_mesh])
        if result is not None and len(result.vertices) > 0:
            return result
    except Exception:
        pass

    return outer_mesh


def _create_bowl(
    outer_radius: float,
    depth: float,
    wall_thickness: float = 3.0,
    sections: int = 64,
    n_profile: int = 30,
) -> trimesh.Trimesh:
    """
    ボウル / 皿形状 (半球ベースの回転体)
    """
    t = np.linspace(0, 1, n_profile)
    # 外側プロファイル: 底面から縁に向かって広がる
    r_outer = outer_radius * np.sin(t * np.pi / 2)
    z_outer = depth * (1 - np.cos(t * np.pi / 2))

    outer_mesh = _create_surface_of_revolution(r_outer, z_outer, sections)

    # 内側を削る
    inner_r = np.maximum(r_outer - wall_thickness, 0.5)
    inner_z = z_outer
    inner_mesh = _create_surface_of_revolution(inner_r, inner_z, sections)
    inner_mesh.apply_translation([0, 0, wall_thickness])

    try:
        result = trimesh.boolean.difference([outer_mesh, inner_mesh])
        if result is not None and len(result.vertices) > 0:
            return result
    except Exception:
        pass

    return outer_mesh


def _create_twisted_prism(
    n_sides: int,
    radius: float,
    height: float,
    twist_angle_deg: float,
    sections: int = 48,
) -> trimesh.Trimesh:
    """
    ねじれた角柱 / スパイラル柱
    n_sides: 断面の辺数 (3=三角, 4=四角, 6=六角 など)
    twist_angle_deg: 上端での合計ねじれ角 (度)
    """
    twist_rad = np.radians(twist_angle_deg)
    base_angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    verts = []
    for layer in range(sections + 1):
        t = layer / sections
        z = height * t
        rot = twist_rad * t
        for a in base_angles:
            verts.append([radius * np.cos(a + rot), radius * np.sin(a + rot), z])

    verts = np.array(verts)
    bc_idx = len(verts)
    tc_idx = len(verts) + 1
    verts = np.vstack([verts, [[0, 0, 0]], [[0, 0, height]]])

    faces = []

    def vi(layer, side):
        return layer * n_sides + side % n_sides

    for layer in range(sections):
        for s in range(n_sides):
            v00, v01 = vi(layer, s), vi(layer, s + 1)
            v10, v11 = vi(layer + 1, s), vi(layer + 1, s + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    for s in range(n_sides):
        faces.append([bc_idx, vi(0, s + 1), vi(0, s)])
    for s in range(n_sides):
        faces.append([tc_idx, vi(sections, s), vi(sections, s + 1)])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int32))
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _create_wavy_plate(
    width: float,
    depth: float,
    base_height: float,
    amplitude: float,
    wave_count: float = 3.0,
    resolution: int = 48,
) -> trimesh.Trimesh:
    """
    波打った表面を持つプレート (正弦波サーフェス)
    """
    n = resolution
    xs = np.linspace(0, width, n)
    ys = np.linspace(0, depth, n)
    X, Y = np.meshgrid(xs, ys)  # shape (n, n)

    Z_top = base_height + amplitude * (
        np.sin(2 * np.pi * wave_count * X / width) *
        np.cos(2 * np.pi * wave_count * Y / depth)
    )

    def ti(i, j):  # top vertex index
        return i * n + j

    def bi(i, j):  # bottom vertex index
        return n * n + i * n + j

    top_verts = np.column_stack([X.ravel(), Y.ravel(), Z_top.ravel()])
    bot_verts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
    all_verts = np.vstack([top_verts, bot_verts])

    faces = []

    # Top surface
    for i in range(n - 1):
        for j in range(n - 1):
            v00, v10, v01, v11 = ti(i, j), ti(i + 1, j), ti(i, j + 1), ti(i + 1, j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    # Bottom surface (reversed winding)
    for i in range(n - 1):
        for j in range(n - 1):
            v00, v10, v01, v11 = bi(i, j), bi(i + 1, j), bi(i, j + 1), bi(i + 1, j + 1)
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])

    # Front wall (j=0)
    for i in range(n - 1):
        faces.append([ti(i, 0), bi(i, 0), bi(i + 1, 0)])
        faces.append([ti(i, 0), bi(i + 1, 0), ti(i + 1, 0)])

    # Back wall (j=n-1)
    for i in range(n - 1):
        faces.append([ti(i, n - 1), ti(i + 1, n - 1), bi(i + 1, n - 1)])
        faces.append([ti(i, n - 1), bi(i + 1, n - 1), bi(i, n - 1)])

    # Left wall (i=0)
    for j in range(n - 1):
        faces.append([ti(0, j), ti(0, j + 1), bi(0, j + 1)])
        faces.append([ti(0, j), bi(0, j + 1), bi(0, j)])

    # Right wall (i=n-1)
    for j in range(n - 1):
        faces.append([ti(n - 1, j), bi(n - 1, j), bi(n - 1, j + 1)])
        faces.append([ti(n - 1, j), bi(n - 1, j + 1), ti(n - 1, j + 1)])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces, dtype=np.int32))
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _create_pyramid(base: float, height: float) -> trimesh.Trimesh:
    """四角錐 (ピラミッド)"""
    half = base / 2.0
    vertices = np.array([
        [-half, -half, 0],
        [half, -half, 0],
        [half, half, 0],
        [-half, half, 0],
        [0, 0, height],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [0, 1, 4], [1, 2, 4],
        [2, 3, 4], [3, 0, 4],
    ], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _create_hollow_cylinder(
    outer_radius: float, inner_radius: float, height: float, sections: int = 64
) -> trimesh.Trimesh:
    """中空円柱 (パイプ)"""
    outer = trimesh.creation.cylinder(radius=outer_radius, height=height, sections=sections)
    inner = trimesh.creation.cylinder(radius=inner_radius, height=height + 0.1, sections=sections)
    try:
        result = trimesh.boolean.difference([outer, inner])
        if result is not None and len(result.vertices) > 0:
            return result
    except Exception:
        pass
    return outer


def get_supported_shapes() -> list:
    """サポートする形状の一覧を返す"""
    return [
        {
            "name": "box",
            "label": "直方体 (Box)",
            "params": ["width (mm)", "depth (mm)", "height (mm)"],
            "example": "幅20mm、奥行き15mm、高さ10mmの直方体",
        },
        {
            "name": "rounded_box",
            "label": "角丸直方体 (Rounded Box)",
            "params": ["width (mm)", "depth (mm)", "height (mm)", "radius (mm)"],
            "example": "幅30mm、高さ10mm、角丸半径2mmのボックス",
        },
        {
            "name": "sphere",
            "label": "球体 (Sphere)",
            "params": ["radius (mm)"],
            "example": "半径25mmの球",
        },
        {
            "name": "hemisphere",
            "label": "半球 (Hemisphere / Dome)",
            "params": ["radius (mm)"],
            "example": "半径20mmのドーム",
        },
        {
            "name": "ellipsoid",
            "label": "楕円体 (Ellipsoid)",
            "params": ["radius_x (mm)", "radius_y (mm)", "radius_z (mm)"],
            "example": "rx=20, ry=15, rz=10の楕円体",
        },
        {
            "name": "cylinder",
            "label": "円柱 (Cylinder)",
            "params": ["radius (mm)", "height (mm)"],
            "example": "半径10mm、高さ50mmの円柱",
        },
        {
            "name": "cone",
            "label": "円錐 (Cone)",
            "params": ["radius (mm)", "height (mm)"],
            "example": "底面半径15mm、高さ40mmの円錐",
        },
        {
            "name": "frustum",
            "label": "台形円錐 (Frustum)",
            "params": ["bottom_radius (mm)", "top_radius (mm)", "height (mm)"],
            "example": "底面半径20mm、上面半径10mm、高さ30mmの切頭円錐",
        },
        {
            "name": "torus",
            "label": "トーラス (Torus)",
            "params": ["major_radius (mm)", "minor_radius (mm)"],
            "example": "外径30mm、管径8mmのドーナツ形",
        },
        {
            "name": "capsule",
            "label": "カプセル (Capsule)",
            "params": ["radius (mm)", "height (mm)"],
            "example": "半径8mm、高さ40mmのカプセル",
        },
        {
            "name": "pyramid",
            "label": "ピラミッド (Pyramid)",
            "params": ["base (mm)", "height (mm)"],
            "example": "底面30mm、高さ40mmのピラミッド",
        },
        {
            "name": "star",
            "label": "星形 (Star)",
            "params": ["points (個)", "outer_radius (mm)", "inner_radius (mm)", "height (mm)"],
            "example": "5角星、外径20mm、高さ8mm",
        },
        {
            "name": "arrow",
            "label": "矢印 (Arrow)",
            "params": ["total_length (mm)", "shaft_width (mm)", "head_width (mm)", "head_length (mm)", "thickness (mm)"],
            "example": "全長40mm、厚み8mmの矢印",
        },
        {
            "name": "cross",
            "label": "十字形 (Cross)",
            "params": ["arm_length (mm)", "arm_width (mm)", "height (mm)"],
            "example": "腕の長さ40mm、幅12mm、高さ8mmの十字",
        },
        {
            "name": "gear",
            "label": "歯車 (Gear)",
            "params": ["teeth (歯数)", "module (モジュール)", "thickness (mm)", "bore_radius (mm)"],
            "example": "歯数16、モジュール2、厚み10mmの歯車",
        },
        {
            "name": "spring",
            "label": "コイルスプリング (Spring)",
            "params": ["coil_radius (mm)", "wire_radius (mm)", "num_coils (巻数)", "pitch (mm)"],
            "example": "コイル半径10mm、線径1.5mm、5巻き",
        },
        {
            "name": "l_bracket",
            "label": "L字ブラケット (L-Bracket)",
            "params": ["arm1_length (mm)", "arm2_length (mm)", "thickness (mm)", "depth (mm)"],
            "example": "縦30mm、横30mm、厚み3mm、奥行き20mmのL字金具",
        },
        {
            "name": "t_bracket",
            "label": "T字ブラケット (T-Bracket)",
            "params": ["top_length (mm)", "stem_length (mm)", "arm_width (mm)", "depth (mm)"],
            "example": "上部40mm、軸30mm、幅8mm、奥行き10mmのT字",
        },
        {
            "name": "pipe",
            "label": "パイプ (Pipe)",
            "params": ["outer_radius (mm)", "inner_radius (mm)", "height (mm)"],
            "example": "外径12mm、内径8mm、長さ50mmのパイプ",
        },
        {
            "name": "hexagonal_prism",
            "label": "六角柱 (Hexagonal Prism)",
            "params": ["radius (mm)", "height (mm)"],
            "example": "内接円半径10mm、高さ20mmの六角柱",
        },
        {
            "name": "wedge",
            "label": "扇形柱 (Wedge/Sector)",
            "params": ["radius (mm)", "angle (度)", "height (mm)"],
            "example": "半径20mm、角度90°、高さ10mmの扇形",
        },
        {
            "name": "compound",
            "label": "複合形状 (Compound)",
            "params": ["components: [{shape, dimensions, position, rotation, operation}]"],
            "example": "穴あきベースプレート、ボルト台座、ロボットのボディなど複数パーツの組み合わせ",
        },
    ]
