"""
STL Generator - テキストパラメータから3Dメッシュを生成してSTLとして出力する
Bambu Lab mini 対応 (標準バイナリSTL形式)
"""

import trimesh
import numpy as np


def generate_stl(shape_params: dict) -> bytes:
    """
    形状パラメータからSTLバイナリデータを生成する

    Args:
        shape_params: Claude APIから返される形状パラメータ
            {
                "shape": "cylinder"|"box"|"sphere"|"cone"|"torus"|"capsule"|"pyramid",
                "dimensions": {形状に応じたパラメータ (mm単位)},
                ...
            }

    Returns:
        バイナリSTLデータ
    """
    shape = shape_params.get("shape", "box").lower()
    dims = shape_params.get("dimensions", {})

    mesh = _create_shape(shape, dims)

    # メッシュをリセット・正規化
    mesh.process(validate=True)

    # Bambu Lab用: メッシュの底面をZ=0に配置
    bounds = mesh.bounds
    if bounds is not None and np.asarray(bounds).ndim == 2:
        z_min = float(bounds[0][2])
        if z_min < 0:
            mesh.apply_translation([0, 0, -z_min])

    # バイナリSTLとして出力
    stl_bytes = mesh.export(file_type="stl")
    return stl_bytes


def _create_shape(shape: str, dims: dict) -> trimesh.Trimesh:
    """形状タイプに応じてtrimeshメッシュを生成する"""

    # --- 直方体 / 立方体 ---
    if shape in ["box", "cube", "直方体", "立方体", "rectangular", "rectangle"]:
        width = float(dims.get("width", dims.get("w", 20)))
        depth = float(dims.get("depth", dims.get("d", 20)))
        height = float(dims.get("height", dims.get("h", 20)))
        return trimesh.creation.box(extents=[width, depth, height])

    # --- 球体 ---
    elif shape in ["sphere", "球", "球体", "ball"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        subdivisions = int(dims.get("subdivisions", 4))
        return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

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
        return trimesh.creation.capsule(radius=radius, height=height, count=count)

    # --- ピラミッド ---
    elif shape in ["pyramid", "ピラミッド"]:
        base = float(dims.get("base", dims.get("width", 20)))
        height = float(dims.get("height", dims.get("h", 20)))
        return _create_pyramid(base, height)

    # --- 中空円柱 (パイプ/チューブ) ---
    elif shape in ["pipe", "hollow_cylinder", "パイプ", "中空円柱"]:
        outer_radius = float(dims.get("outer_radius", dims.get("radius", 12)))
        inner_radius = float(dims.get("inner_radius", dims.get("inner_r", 8)))
        height = float(dims.get("height", dims.get("h", 30)))
        sections = int(dims.get("sections", 64))
        return _create_hollow_cylinder(outer_radius, inner_radius, height, sections)

    # --- 六角柱 ---
    elif shape in ["hexagonal_prism", "六角柱", "hex_prism"]:
        radius = float(dims.get("radius", dims.get("r", 10)))
        height = float(dims.get("height", dims.get("h", 20)))
        return trimesh.creation.cylinder(radius=radius, height=height, sections=6)

    # --- デフォルト: 直方体 ---
    else:
        return trimesh.creation.box(extents=[20, 20, 20])


def _create_pyramid(base: float, height: float) -> trimesh.Trimesh:
    """四角錐（ピラミッド）を生成する"""
    half = base / 2.0
    vertices = np.array(
        [
            [-half, -half, 0],  # 底面4頂点
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
            [0, 0, height],  # 頂点
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],  # 底面三角形1
            [0, 2, 3],  # 底面三角形2
            [0, 1, 4],  # 側面
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def _create_hollow_cylinder(
    outer_radius: float, inner_radius: float, height: float, sections: int = 64
) -> trimesh.Trimesh:
    """中空円柱（パイプ）を生成する"""
    outer = trimesh.creation.cylinder(
        radius=outer_radius, height=height, sections=sections
    )
    inner = trimesh.creation.cylinder(
        radius=inner_radius, height=height + 0.1, sections=sections
    )
    # ブーリアン差分で中空にする
    try:
        result = outer.difference(inner)
        return result
    except Exception:
        # ブーリアン演算が失敗した場合は外側の円柱を返す
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
            "name": "sphere",
            "label": "球体 (Sphere)",
            "params": ["radius (mm)"],
            "example": "半径25mmの球",
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
    ]
