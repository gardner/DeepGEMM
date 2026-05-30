from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def workflow_header(path: str) -> str:
    return "\n".join(read(path).splitlines()[:24])


def test_gb10_release_workflow_builds_native_sm121a_wheel():
    workflow = read(".github/workflows/gb10-release.yml")

    assert "gb10-deepgemm-v*" in workflow
    assert 'TORCH_CUDA_ARCH_LIST: "12.1a"' in workflow
    assert 'CMAKE_CUDA_ARCHITECTURES: "121a"' in workflow
    assert "torch-wheel-url" in workflow
    assert "cuobjdump" in workflow
    assert "gh release upload" in workflow


def test_upstream_publish_matrix_is_manual_only_for_the_fork():
    header = workflow_header(".github/workflows/publish.yml")

    assert "workflow_dispatch:" in header
    assert "create:" not in header
