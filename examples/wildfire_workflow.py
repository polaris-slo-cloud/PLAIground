"""Workflow-based Wildfire Detection Example.

This example demonstrates the class-based declarative workflow:
- @selector decorator on class specifies model selection strategy
- @compoundable_model on methods declares tasks
- run() method defines orchestration logic

Run with:
    python -m v2.examples.wildfire_workflow
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Core imports
from v2.core.decorator import compoundable_model
from v2.workflow import DeclarativeWorkflow, selector
from v2.selector import AdaptiveSelector, SLOConstraints


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class SatelliteImage:
    """Input: satellite/drone image data."""
    image_path: str
    coordinates: Optional[tuple] = None
    timestamp: Optional[str] = None


@dataclass
class FireDetection:
    """Output: fire detection result."""
    has_fire: bool
    confidence: float
    detections: List[dict]


@dataclass
class FireReport:
    """Output: generated fire report."""
    summary: str
    recommendations: List[str]

@selector(
    AdaptiveSelector,
    slo=SLOConstraints(
        min_accuracy=0.85,
        max_p95_latency_ms=3000,
        max_total_cost=0.50
    ),
    total_requests=50,
    adaptation_interval=5
)
class WildfireDetectionWorkflow(DeclarativeWorkflow):

    @compoundable_model(capability="object_detection")
    def detect_fire(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @compoundable_model(
        capability="llm",
        task_config={
            "prompt": (
                "You are a wildfire incident analyst. Generate a detailed fire incident report.\n\n"
                "Location: {location}\n"
                "Detection confidence: {confidence}\n"
                "Number of fire zones detected: {num_detections}\n\n"
                "Provide a summary of the incident and recommended actions."
            )
        }
    )
    def generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def run(self, image: SatelliteImage) -> FireReport:

        detection_result = self.detect_fire({"image": image.image_path})

        has_fire = detection_result.get("has_detection", False)
        detections = detection_result.get("detections", [])
        confidence = max((d.get("confidence", 0) for d in detections), default=0)

        if has_fire:
            # Just pass the data - Task Configuration has the prompt template
            report_result = self.generate_report({
                "location": str(image.coordinates),
                "confidence": f"{confidence:.0%}",
                "num_detections": len(detections)
            })

            return FireReport(
                summary=report_result.get("output", "Fire detected"),
                recommendations=["Dispatch fire response team", "Evacuate nearby areas"]
            )
        else:
            return FireReport(
                summary="No fire detected",
                recommendations=["Continue monitoring"]
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("WILDFIRE DETECTION WORKFLOW (Class-based)")
    print("=" * 60)

    # Create workflow instance - auto-configures selector and binds models
    workflow = WildfireDetectionWorkflow()

    # Create test images
    test_images = [
        SatelliteImage(
            image_path="/data/satellite/region_42.tif",
            coordinates=(45.123, -122.456),
            timestamp="2024-01-15T14:30:00Z"
        ),
        SatelliteImage(
            image_path="/data/satellite/region_43.tif",
            coordinates=(45.200, -122.500),
            timestamp="2024-01-15T14:35:00Z"
        ),
        SatelliteImage(
            image_path="/data/satellite/region_44.tif",
            coordinates=(45.300, -122.600),
            timestamp="2024-01-15T14:40:00Z"
        ),
    ]

    # Process images
    print("\nProcessing satellite images...\n")
    for i, image in enumerate(test_images):
        print(f"[Image {i+1}] {image.image_path}")
        report = workflow.run(image)
        print(f"  Summary: {report.summary}")
        print(f"  Recommendations: {report.recommendations}")
        print()

    # Show workflow summary
    summary = workflow.get_summary()
    print("=" * 60)
    print("WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"  Total requests: {workflow.get_selector().requests_processed}")
    print(f"  Accuracy: {summary['final_accuracy']:.1%}")
    print(f"  Total cost: ${summary['total_cost']:.4f}")
    print(f"  Avg latency: {summary['avg_latency']:.0f}ms")
    print(f"  Model switches: {summary['total_switches']}")
    print(f"  SLO compliant: {summary['slo_compliant']}")


if __name__ == "__main__":
    main()
