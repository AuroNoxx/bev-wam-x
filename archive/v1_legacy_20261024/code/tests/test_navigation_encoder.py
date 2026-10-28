import torch
from models.baseline import ResNet18Trajectory
from models.navigation_encoder import ResNetNavigationEncoder


def test_resnet_navigation_features_preserve_original_spatial_features():
    torch.manual_seed(91)
    baseline = ResNet18Trajectory().eval()
    encoder = ResNetNavigationEncoder(baseline).eval()
    bev = torch.randint(0, 256, (2, 4, 128, 128), dtype=torch.uint8)
    with torch.no_grad():
        old = baseline.pool(baseline.backbone(baseline.stem(bev.float()/255.)))
        actual = encoder(bev)
    torch.testing.assert_close(actual.transpose(1, 2).reshape_as(old), old, atol=0, rtol=0)
    assert actual.shape == (2, 16, 512)
    assert not any(k.startswith('head.') for k in encoder.state_dict())
