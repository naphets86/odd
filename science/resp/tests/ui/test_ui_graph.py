"""Graph analysis page – subgraph comparison and redundancy check via UI."""
import httpx
from playwright.sync_api import Page, expect


def _goto_graph(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="graph"]')
    page.wait_for_load_state("networkidle")


def _create_project_with_person(
    api: httpx.Client, project_name: str, person_name: str
) -> tuple[int, int]:
    proj_id = api.post(
        "/api/projects/", json={"name": project_name, "status": "aktiv"}
    ).json()["id"]
    pers_id = api.post(
        "/api/persons/", json={"name": person_name, "person_type": "Arbeitnehmer"}
    ).json()["id"]
    api.post(
        "/api/allocations/",
        json={"project_id": proj_id, "person_id": pers_id, "quantity": 1, "status": "aktiv"},
    )
    return proj_id, pers_id


def test_graph_page_loads(page: Page, base_url: str) -> None:
    _goto_graph(page, base_url)
    expect(page.locator("#page-graph")).to_be_visible()
    expect(page.locator("button:has-text('Vergleichen')")).to_be_visible()
    expect(page.locator("button:has-text('Redundanzcheck')")).to_be_visible()


def test_graph_selects_populated_with_projects(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    api.post("/api/projects/", json={"name": "Graph Projekt X", "status": "aktiv"})
    _goto_graph(page, base_url)
    expect(page.locator("#graph-proj-a")).to_contain_text("Graph Projekt X")
    expect(page.locator("#graph-proj-b")).to_contain_text("Graph Projekt X")


def test_graph_compare_no_selection_shows_error(page: Page, base_url: str) -> None:
    _goto_graph(page, base_url)
    page.click("button:has-text('Vergleichen')")
    expect(page.locator("#toast-container")).to_contain_text("mindestens ein Projekt")


def test_graph_compare_shows_result(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    proj_a_id, _ = _create_project_with_person(api, "Graph A Projekt", "Person A")
    proj_b_id, _ = _create_project_with_person(api, "Graph B Projekt", "Person B")

    _goto_graph(page, base_url)
    page.select_option("#graph-proj-a", str(proj_a_id))
    page.select_option("#graph-proj-b", str(proj_b_id))
    page.click("button:has-text('Vergleichen')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#graph-result")).to_be_visible()
    # Decision badge must be present
    expect(page.locator("#graph-result .badge")).to_be_visible()
    # Bipartite visualisation card appears
    expect(page.locator("#graph-viz-card")).to_be_visible()


def test_graph_compare_toast(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    proj_a_id, _ = _create_project_with_person(api, "Toast A", "Person TA")
    proj_b_id, _ = _create_project_with_person(api, "Toast B", "Person TB")

    _goto_graph(page, base_url)
    page.select_option("#graph-proj-a", str(proj_a_id))
    page.select_option("#graph-proj-b", str(proj_b_id))
    page.click("button:has-text('Vergleichen')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#toast-container")).to_contain_text("Entscheidung:")


def test_graph_redundancy_requires_two_projects(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    api.post("/api/projects/", json={"name": "Einzel Projekt", "status": "aktiv"})
    _goto_graph(page, base_url)
    page.click("button:has-text('Redundanzcheck')")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#toast-container")).to_contain_text("Mindestens 2 Projekte")


def test_graph_redundancy_shows_result(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    _create_project_with_person(api, "Redund A", "Pers R1")
    _create_project_with_person(api, "Redund B", "Pers R2")

    _goto_graph(page, base_url)
    page.click("button:has-text('Redundanzcheck')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#graph-result")).to_be_visible()
