# ruff: noqa: E402
import sys
sys.path.insert(0, '/app')

# Force-register custom transformers into __main__ so joblib can unpickle them
from src.features.preprocess import (
    IsGustyTransformer,
    NumCloudLayersTransformer,
    CloudCeilingTransformer,
    WxCodeTransformer,
    SkyC1Encoder,
)
import __main__
__main__.IsGustyTransformer = IsGustyTransformer
__main__.NumCloudLayersTransformer = NumCloudLayersTransformer
__main__.CloudCeilingTransformer = CloudCeilingTransformer
__main__.WxCodeTransformer = WxCodeTransformer
__main__.SkyC1Encoder = SkyC1Encoder

# Now start uvicorn
import uvicorn
uvicorn.run('backend.main:app', host='0.0.0.0', port=8000)
