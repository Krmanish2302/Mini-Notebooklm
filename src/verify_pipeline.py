# Verification script for each component
def verify_component(component_name, import_path, test_func):
    try:
        module = __import__(import_path, fromlist=[component_name])
        component = getattr(module, component_name)
        result = test_func(component)
        print(f"✅ {component_name}: PASSED")
        return True
    except Exception as e:
        print(f"❌ {component_name}: FAILED - {e}")
        return False

# Example verifications
verify_component("FileDetector", "src.ingestion.file_detector", 
                lambda x: x.detect(file_path="test.pdf"))
verify_component("PDFPipeline", "src.ingestion.pipelines.pdf_pipeline",
                lambda x: x.process("test.pdf", "source_1"))
verify_component("MasterPipeline", "src.master_pipeline",
                lambda x: x.get_stats())