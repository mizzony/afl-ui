"""Team and venue reference data.

Canonical team names follow the Squiggle API, so live data and the bundled
history line up without extra mapping.
"""

TEAM_STATE = {
    "Adelaide": "SA",
    "Brisbane Lions": "QLD",
    "Carlton": "VIC",
    "Collingwood": "VIC",
    "Essendon": "VIC",
    "Fremantle": "WA",
    "Geelong": "VIC",
    "Gold Coast": "QLD",
    "Greater Western Sydney": "NSW",
    "Hawthorn": "VIC",
    "Melbourne": "VIC",
    "North Melbourne": "VIC",
    "Port Adelaide": "SA",
    "Richmond": "VIC",
    "St Kilda": "VIC",
    "Sydney": "NSW",
    "West Coast": "WA",
    "Western Bulldogs": "VIC",
}

TEAMS = sorted(TEAM_STATE)

SHORT_NAME = {
    "Greater Western Sydney": "GWS",
    "Western Bulldogs": "Bulldogs",
    "Brisbane Lions": "Brisbane",
    "North Melbourne": "North Melb",
    "Port Adelaide": "Port Adel",
}

TEAM_ALIASES = {
    "Kangaroos": "North Melbourne",
    "North Melb": "North Melbourne",
    "GWS": "Greater Western Sydney",
    "GWS Giants": "Greater Western Sydney",
    "Brisbane": "Brisbane Lions",
    "Footscray": "Western Bulldogs",
    "Bulldogs": "Western Bulldogs",
    "Sydney Swans": "Sydney",
    "Geelong Cats": "Geelong",
    "Gold Coast Suns": "Gold Coast",
    "West Coast Eagles": "West Coast",
    "Adelaide Crows": "Adelaide",
}

# Venue -> state. Names follow AFL Tables / Squiggle, plus common sponsor names.
VENUE_STATE = {
    "M.C.G.": "VIC", "MCG": "VIC",
    "Docklands": "VIC", "Marvel Stadium": "VIC", "Etihad Stadium": "VIC",
    "Kardinia Park": "VIC", "GMHBA Stadium": "VIC",
    "Princes Park": "VIC", "Eureka Stadium": "VIC", "Mars Stadium": "VIC",
    "Subiaco": "WA", "Perth Stadium": "WA", "Optus Stadium": "WA", "W.A.C.A.": "WA",
    "Gabba": "QLD", "Carrara": "QLD", "People First Stadium": "QLD",
    "Heritage Bank Stadium": "QLD", "Metricon Stadium": "QLD",
    "Cazaly's Stadium": "QLD", "Riverway Stadium": "QLD",
    "Football Park": "SA", "Adelaide Oval": "SA", "Norwood Oval": "SA",
    "Barossa Park": "SA", "Adelaide Hills": "SA",
    "S.C.G.": "NSW", "SCG": "NSW", "Sydney Showground": "NSW",
    "ENGIE Stadium": "NSW", "Giants Stadium": "NSW",
    "Stadium Australia": "NSW", "Blacktown": "NSW",
    "Manuka Oval": "ACT", "Corroboree Group Oval Manuka": "ACT",
    "York Park": "TAS", "UTAS Stadium": "TAS", "Bellerive Oval": "TAS",
    "Blundstone Arena": "TAS", "Ninja Stadium": "TAS",
    "Marrara Oval": "NT", "TIO Stadium": "NT", "Traeger Park": "NT",
    "TIO Traeger Park": "NT",
    "Wellington": "NZ", "Jiangwan Stadium": "CHN", "Summit Sports Park": "CHN",
}

# Teams that treat a venue outside their state as a regular "home" ground.
SECOND_HOMES = {
    ("Hawthorn", "York Park"),
    ("North Melbourne", "Bellerive Oval"),
    ("Greater Western Sydney", "Manuka Oval"),
    ("Melbourne", "Traeger Park"),
}


def canonical_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def short(name: str) -> str:
    return SHORT_NAME.get(name, name)


def venue_state(venue: str, home_team: str) -> str:
    """State of a venue. Unknown venues are assumed to be in the home team's state."""
    return VENUE_STATE.get(str(venue).strip(), TEAM_STATE.get(home_team, "VIC"))


def travel_edge(home: str, away: str, venue: str) -> int:
    """+1 if only the away team travels interstate, -1 if only the home team does, else 0."""
    vs = venue_state(venue, home)
    home_local = TEAM_STATE.get(home) == vs or (home, venue) in SECOND_HOMES
    away_local = TEAM_STATE.get(away) == vs or (away, venue) in SECOND_HOMES
    return int(home_local) - int(away_local)
