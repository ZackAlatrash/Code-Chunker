#!/usr/bin/env python3
"""
Chunk Doctor Test Runner

Simple script to run the complete Chunk Doctor testing pipeline.
"""

import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the Chunk Doctor test pipeline."""
    logger.info("🚀 Starting Chunk Doctor Test Pipeline")
    
    try:
        # Run the main test pipeline
        result = subprocess.run([
            "python", "test_chunk_doctor_pipeline.py"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✅ Test pipeline completed successfully")
        print(result.stdout)
        
        # Run quality analysis
        logger.info("📊 Running quality analysis...")
        result = subprocess.run([
            "python", "analyze_chunk_quality.py", 
            "tests/goldens/go/weather_forecast_service.jsonl"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✅ Quality analysis completed")
        print(result.stdout)
        
        logger.info("🎉 All tests passed! Chunk Doctor pipeline is working correctly.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Test failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
