# pose_estimation/models/model_manager.py
import openvino as ov
import openvino.properties.hint as hints
from pathlib import Path
import requests


class ModelManager:
    """OpenVINO 모델 관리 클래스"""
    
    def __init__(self, base_model_dir="model"):
        self.base_model_dir = Path(base_model_dir)
        self.core = ov.Core()
        self.model = None
        self.compiled_model = None
        
    def download_model(self, model_name="human-pose-estimation-0001", precision="FP16-INT8"):
        """모델 다운로드"""
        model_path = self.base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"
        
        if not model_path.exists():
            model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
            
            # 디렉토리 생성
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # XML 파일 다운로드
            xml_response = requests.get(model_url_dir + model_name + ".xml")
            with open(model_path, 'wb') as f:
                f.write(xml_response.content)
                
            # BIN 파일 다운로드
            bin_response = requests.get(model_url_dir + model_name + ".bin")
            with open(model_path.with_suffix(".bin"), 'wb') as f:
                f.write(bin_response.content)
                
        return model_path
    
    def load_model(self, model_path, device_name="AUTO"):
        """모델 로드 및 컴파일"""
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(
            model=self.model, 
            device_name=device_name,
            config={hints.performance_mode(): hints.PerformanceMode.LATENCY}
        )
        return self.compiled_model
    
    def get_input_output_info(self):
        """입력/출력 정보 반환"""
        if self.compiled_model is None:
            raise ValueError("Model not loaded. Call load_model first.")
            
        input_layer = self.compiled_model.input(0)
        output_layers = self.compiled_model.outputs
        height, width = list(input_layer.shape)[2:]
        
        return {
            'input_layer': input_layer,
            'output_layers': output_layers,
            'height': height,
            'width': width,
            'input_name': input_layer.any_name,
            'output_names': [o.any_name for o in output_layers]
        }


