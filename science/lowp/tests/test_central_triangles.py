import math

import pytest

from src import central_triangles as ct


def test_sector_subdivision_and_round_counts():
    assert ct.sector_angle(3) == pytest.approx(math.pi / 4)
    assert ct.sector_count(3) == 8
    assert ct.rounds_for_parts(8) == 3
    sectors = ct.sectors_after(3)
    assert len(sectors) == 8
    for (phi0, alpha), k in zip(sectors, range(8)):
        assert phi0 == pytest.approx(k * math.pi / 4)
        assert alpha == pytest.approx(math.pi / 4)
    left, right = ct.split_sector(0.2, 1.4)
    assert left == pytest.approx((0.2, 0.7))
    assert right == pytest.approx((0.9, 0.7))


@pytest.mark.parametrize("n", [3, 4, 8, 12])
def test_regular_polygon_geometry(n):
    assert len(ct.vertices(n)) == n
    assert abs(ct.vertex(0, n)) == pytest.approx(1.0)
    assert ct.vertex(n, n) == pytest.approx(ct.vertex(0, n))
    assert ct.base_length(n) == pytest.approx(2 * math.sin(math.pi / n))
    assert ct.triangle_area(n) == pytest.approx(0.5 * math.sin(2 * math.pi / n))
    assert ct.foot_point(n - 1, n) == pytest.approx(ct.foot_point_polar(n - 1, n))
    assert ct.rotate_triangle(ct.triangle_corners(0, n), n) == pytest.approx(
        ct.triangle_corners(1, n)
    )


def test_octagon_exact_values_and_areas():
    exact = ct.exact_values()
    assert exact["sin_pi_8"] == pytest.approx(math.sin(math.pi / 8))
    assert exact["cos_pi_8"] == pytest.approx(math.cos(math.pi / 8))
    assert exact["tan_pi_8"] == pytest.approx(math.tan(math.pi / 8))
    assert ct.polygon_area() == pytest.approx(2 * math.sqrt(2))
    assert ct.octagon_area() == pytest.approx(ct.polygon_area())
    assert ct.fill_ratio() == pytest.approx(ct.polygon_area() / math.pi)
    assert ct.angle_sums()["apex_total"] == pytest.approx(2 * math.pi)
    assert ct.angle_sums()["all"] == pytest.approx(8 * math.pi)


def test_triangle_coordinates_membership_and_tiling():
    z = 0.2 * ct.vertex(2) + 0.3 * ct.vertex(3)
    assert ct.triangle_coordinates(z, 2) == pytest.approx((0.2, 0.3))
    assert ct.in_cone(z, 2)
    assert ct.in_triangle(z, 2)
    assert ct.locate(z) == [2]
    assert ct.locate(0j) == list(range(8))
    assert ct.shared_edge_vertices(0, 1) == [1]
    assert ct.shared_edge_vertices(0, 4) == []


def test_polygon_membership_and_pi_bounds():
    assert ct.in_polygon(0j)
    assert ct.in_polygon(ct.vertex(0))
    assert not ct.in_polygon(2 + 0j)
    assert ct.in_circumscribed_polygon(1 + 0j)
    assert not ct.in_circumscribed_polygon(2 + 0j)
    lower, upper = ct.pi_bounds(8)
    assert lower < math.pi < upper
    assert ct.inscribed_area(8) == pytest.approx(lower)
    assert ct.circumscribed_area(8) == pytest.approx(upper)


def test_regular_spacing_congruence_and_uniqueness():
    regular = [2 * math.pi * k / 8 for k in range(8)]
    irregular = [0.0, 0.4, 1.1, 2.3]
    assert ct.central_angles(regular) == pytest.approx([math.pi / 4] * 8)
    assert ct.triangles_congruent(regular)
    assert ct.uniqueness_gap(regular) == pytest.approx(math.pi / 4)
    assert not ct.triangles_congruent(irregular)
    assert ct.uniqueness_gap(irregular) is None


def test_area_bound_refinement_and_richardson():
    rows = ct.refine_bounds(2, 4)
    assert [row[0] for row in rows] == [4, 8, 16, 32, 64]
    for _, lower, upper in rows:
        assert lower < math.pi < upper
    assert all(rows[i + 1][2] - rows[i + 1][1] < rows[i][2] - rows[i][1]
               for i in range(len(rows) - 1))
    assert ct.richardson_combination(8) == pytest.approx(ct.richardson_closed_form_8())
    assert ct.doubled_areas(2, 4)[0] == pytest.approx(math.sqrt(8))


def test_arctangent_and_central_triangle_series_bounds():
    for n_terms in (1, 3, 10):
        partial = ct.arctan_partial(1.0, n_terms)
        assert abs(math.pi / 4 - partial) <= ct.arctan_remainder_bound(1.0, n_terms)
        value = ct.central_triangle_series(8, n_terms)
        assert abs(math.pi - value) <= ct.central_triangle_error_bound(8, n_terms)
        assert ct.central_triangle_error_sign(n_terms) == (-1 if n_terms % 2 else 1)
    assert ct.leibniz_partial(500) == pytest.approx(math.pi, abs=0.003)
    assert ct.octant_partial(10) == pytest.approx(math.pi, abs=0.01)
    left, right = ct.leibniz_octant_identity(1000)
    assert left == pytest.approx(math.pi / 4, abs=0.001)
    assert right == pytest.approx(math.pi / 4, abs=1e-10)
    assert ct.double_angle_identity_holds()


def test_tau_recursion_and_term_bound():
    seq = ct.tau_sequence(6)
    assert seq == pytest.approx([ct.tau_exact(k) for k in range(7)])
    assert ct.tau_degree_bound(4) == 16
    for k in range(4):
        n_terms = ct.terms_needed(k, 1e-3)
        assert ct.pi_from_tau(k, n_terms) == pytest.approx(math.pi, abs=1e-3)
        if n_terms > 1:
            previous_bound = 2 ** (k + 2) * ct.tau_exact(k) ** (2 * n_terms - 1) / (2 * n_terms - 1)
            assert previous_bound >= 1e-3


@pytest.mark.parametrize(
    "call",
    [
        lambda: ct.sector_count(-1),
        lambda: ct.rounds_for_parts(3),
        lambda: ct.vertex(0, 2),
        lambda: ct.central_angles([0.0, 1.0]),
        lambda: ct.central_angles([0.0, 2 * math.pi, 3.0]),
        lambda: ct.arctan_partial(1.1, 2),
        lambda: ct.central_triangle_series(3, 2),
        lambda: ct.terms_needed(1, 0.0),
    ],
)
def test_invalid_geometry_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
