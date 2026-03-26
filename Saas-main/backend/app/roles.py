COMMON_FUNCTIONAL_NEEDS = [
    "Upload and manage multiple time series",
    "Preprocess data automatically (missing values and outliers)",
    "Authenticate securely and access role-based features"
]

ROLE_FUNCTIONAL_NEEDS = {
    "energy_grid_operator": [
        "Generate prediction intervals for 48-hour horizon",
        "Trigger alerts when predicted demand exceeds capacity thresholds",
        "Display historical forecasts vs. actual consumption",
        "Provide real-time dashboard with automatic updates"
    ],
    "energy_trader": [
        "Generate multiple quantile levels for risk assessment",
        "Allow custom quantile selection",
        "Export forecast data"
    ],
    "energy_planner": [
        "Generate long-term forecast scenarios with uncertainty",
        "Access trend analysis and seasonality reports",
        "View strategic planning dashboards",
        "Export planning reports"
    ],
    "system_administrator": [
        "Manage user accounts and assign roles",
        "Configure system settings",
        "Access audit logs and security events"
    ],
}


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def get_role_capabilities(role: str) -> list[str]:
    if role == "system_administrator":
        all_actor_specific: list[str] = []
        for actor_role, needs in ROLE_FUNCTIONAL_NEEDS.items():
            if actor_role != "system_administrator":
                all_actor_specific.extend(needs)

        return _unique_in_order(
            ROLE_FUNCTIONAL_NEEDS["system_administrator"]
            + all_actor_specific
            + COMMON_FUNCTIONAL_NEEDS
        )

    return _unique_in_order(ROLE_FUNCTIONAL_NEEDS.get(role, []) + COMMON_FUNCTIONAL_NEEDS)


def get_all_role_capabilities() -> dict[str, list[str]]:
    return {role: get_role_capabilities(role) for role in ROLE_FUNCTIONAL_NEEDS.keys()}
