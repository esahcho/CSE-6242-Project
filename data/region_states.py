REGION_STATES = {
    "Pacific Coast": ["CA"],
    "South": ["FL", "GA", "AL", "MS", "LA", "AR", "SC", "NC", "TN", "VA", "KY", "WV"],
    "Desert Southwest": ["AZ", "NM", "TX", "OK", "NM"],
    "Mountain": ["CO", "UT", "WY", "MT", "NV", "ID"],
    "Northeast": ["NY", "PA", "NJ", "CT", "MA", "RI", "VT", "NH", "ME", "MD", "DE"],
    "Midwest": ["OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "Pacific Northwest": ["WA", "OR"],
}

REGION_COLORS = {
    "Pacific Coast":    "#0077B6",
    "South":            "#E63946",
    "Desert Southwest": "#F4A261",
    "Mountain":         "#2A9D8F",
    "Northeast":        "#457B9D",
    "Midwest":          "#6A994E",
    "Pacific Northwest":"#264653",
}

CITY_REGIONS = {
 'Phoenix': {'state': 'AZ', 'region': 'Desert Southwest'},
 'Los Angeles': {'state': 'CA', 'region': 'Pacific Coast'},
 'Atlanta': {'state': 'GA', 'region': 'South'},
 'Chicago': {'state': 'IL', 'region': 'Midwest'},
 'Boston': {'state': 'MA', 'region': 'Northeast'},
 'Denver': {'state': 'CO', 'region': 'Mountain'},
 'Seattle': {'state': 'WA', 'region': "Pacific Northwest"}
}

CITY_MAP = {
    "Pacific Coast": "Los Angeles",
    "South": "Atlanta",
    "Desert Southwest": "Phoenix",
    "Mountain": "Denver",
    "Northeast": "Boston",
    "Midwest": "Chicago",
    "Pacific Northwest": "Seattle"
}
