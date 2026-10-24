"""Road damage video inference and training helpers."""

from .vehicle_impact import (
    DamageAssessment,
    SpotLocation,
    VehicleImpact,
    WarningAlert,
    analyze_damaged_spot,
    draw_vehicle_warning_overlay,
    summarize_overall_assessment,
)
