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
