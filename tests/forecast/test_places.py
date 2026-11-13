import pytest

from forecast.places import PlaceIndex, norm, parse_coordinates
from forecast.service import ForecastError


@pytest.fixture
def index(places_db):
    return PlaceIndex(places_db)


def names(results):
    return [p.display for p in results]


def test_norm():
    assert norm("Díli") == norm("DILI") == "dili"
    assert norm("São Paulo") == "sao paulo"
    assert norm("St. Louis") == norm("St Louis") == "st louis"
    assert norm("N’Djamena") == norm("N'Djamena") == "n djamena"
    assert norm("Straße") == "strasse"
    assert norm("Île-de-France") == "ile de france"
    hindi = "नई दिल्ली"
    assert norm(hindi) == hindi  # vowel signs are combining marks there, and carry meaning


def test_parse_coordinates():
    assert parse_coordinates("-8.556, 125.56") == (-8.556, 125.56)
    assert parse_coordinates("48.85 2.35") == (48.85, 2.35)
    assert parse_coordinates("48.85N, 2.35W") == (48.85, -2.35)
    assert parse_coordinates("8.5°S 125.5°E") == (-8.5, 125.5)
    assert parse_coordinates("Paris") is None
    assert parse_coordinates("10 Downing Street") is None


def test_same_name_ranks_by_size_and_kind(index):
    results = index.search("Paris")
    assert names(results)[:3] == ["Paris, Île-de-France, France", "Paris, Texas, United States",
                                  "Paris, Tennessee, United States"]
    assert results[0].kind == "capital city" and results[0].match == "exact"
    assert index.search("Same")[0].kind == "village, regional capital"
    assert "Paris Creek" not in " ".join(names(results))  # streams aren't indexed
    assert results[-1].name == "Parisot" and results[-1].match == "prefix"


def test_region_and_country_narrow_the_match(index):
    assert names(index.search("Paris, Texas"))[0] == "Paris, Texas, United States"
    assert names(index.search("paris, tx"))[0] == "Paris, Texas, United States"
    assert names(index.search("Paris, USA")) == ["Paris, Texas, United States", "Paris, Tennessee, United States"]
    assert names(index.search("Paris TX"))[0] == "Paris, Texas, United States"  # no comma: trailing words
    assert names(index.search("paris france"))[0] == "Paris, Île-de-France, France"
    assert names(index.search("Same, Timor-Leste")) == ["Same, Manufahi, Timor-Leste"]
    assert names(index.search("Same, Timor Leste")) == ["Same, Manufahi, Timor-Leste"]  # GeoNames' spelling
    assert names(index.search("Washington, D.C."))[0] == "Washington, United States"
    assert names(index.search("Same, East Timor")) == ["Same, Manufahi, Timor-Leste"]
    assert names(index.search("Same, Tanzania")) == ["Same, Kilimanjaro, Tanzania"]
    assert index.search("Paris, Portugal") == []


def test_alternate_and_accented_names(index):
    lisbon = index.search("Lisboa")
    assert names(lisbon) == ["Lisbon, Portugal"] and lisbon[0].match == "alternate"
    assert names(index.search("Lisbon"))[:2] == ["Lisbon, Portugal", "Lisbon, Ohio, United States"]
    assert names(index.search("dili"))[0] == names(index.search("Díli"))[0] == "Dili, Timor-Leste"
    assert names(index.search("Dili"))[1] == "Delhi, India"  # 'Dili' is also one of Delhi's names
    assert "Ar Raqqah, Raqqa, Syria" not in names(index.search("Nice"))  # no prefixes of alternate names
    assert names(index.search("Sao Paulo"))[0] == "São Paulo, Brazil"
    assert names(index.search("Париж"))[0] == "Paris, Île-de-France, France"
    assert index.search("Old Dili") == []  # historical places are dropped


def test_towns_beat_divisions_named_after_them(index):
    assert [(p.display, p.kind) for p in index.search("Paris")[:2]] == [
        ("Paris, Île-de-France, France", "capital city"), ("Paris, Texas, United States", "town")]
    assert [(p.display, p.kind) for p in index.search("Dili")[:2]] == [
        ("Dili, Timor-Leste", "capital city"), ("Delhi, India", "city, regional capital")]
    assert index.search("Dili Municipality")[0].kind == "region"  # still found by its own name


def test_areas(index):
    sea = index.search("Mediterranean")[0]
    assert sea.display == "Mediterranean Sea" and sea.kind == "sea" and sea.is_area and sea.match == "prefix"
    france = index.search("France")[0]
    assert france.display == "France" and france.kind == "country" and france.is_area
    assert france.match == "exact" and index.search("Republic of France")[0].display == "France"


def test_nearest(index):
    near = index.nearest(-8.556, 125.560)
    assert near[0].name == "Dili" and near[0].distance_km == pytest.approx(1.5, abs=0.5)
    assert index.nearest(-8.5617, 125.6000, limit=1)[0].name == "Becora"
    assert index.nearest(-8.5617, 125.6000, limit=1, min_population=1000)[0].name == "Dili"
    assert index.at(-8.5617, 125.6000).nearby.name == "Dili"  # forecasts name a town, not the hamlet next door
    assert index.nearest(0.0, -140.0) == []  # open ocean
    assert names(index.search("-8.556, 125.560", limit=2)) == ["Dili, Timor-Leste", "Becora, Dili, Timor-Leste"]
    with pytest.raises(ForecastError):
        index.nearest(95, 0)


def test_resolve(index):
    r = index.resolve("Paris")
    assert r.place.display == "Paris, Île-de-France, France" and (r.lat, r.lon) == (48.85341, 2.3488)
    assert [p.display for p in r.alternatives] == ["Paris, Texas, United States", "Paris, Tennessee, United States"]
    assert index.resolve("Mediterranean").alternatives == []
    at = index.resolve("-8.556, 125.560")
    assert at.place is None and at.nearby.name == "Dili" and (at.lat, at.lon) == (-8.556, 125.56)
    with pytest.raises(ForecastError) as e:
        index.resolve("Atlantis")
    assert e.value.status == 404 and "Atlantis" in e.value.message


def test_missing_index(tmp_path):
    index = PlaceIndex(tmp_path / "none.sqlite")
    assert not index.available
    with pytest.raises(ForecastError) as e:
        index.search("Paris")
    assert e.value.status == 503
    assert index.at(1.0, 2.0).nearby is None  # coordinates still work


def test_rebuild_is_picked_up(places_db, tmp_path):
    import shutil

    path = tmp_path / "places.sqlite"
    shutil.copy(places_db, path)
    index = PlaceIndex(path)
    assert index.search("Paris")
    path.unlink()
    with pytest.raises(ForecastError):
        index.search("Paris")
    shutil.copy(places_db, path)
    assert index.search("Paris")
