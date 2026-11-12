from forecast.sources.gfs import (
    FcstSpec,
    forecast_hours,
    merge_ranges,
    parse_fcst,
    parse_idx,
    select_record,
)

# Real lines from gfs.t00z.pgrb2.0p25 .idx files (offsets trimmed to a few records).
F007 = """\
581:407306589:d=2026111100:TMP:2 m above ground:7 hour fcst:
582:408500000:d=2026111100:SPFH:2 m above ground:7 hour fcst:
583:409364363:d=2026111100:DPT:2 m above ground:7 hour fcst:
596:418462594:d=2026111100:APCP:surface:6-7 hour acc fcst:
597:418694253:d=2026111100:APCP:surface:0-7 hour acc fcst:
636:435969922:d=2026111100:TCDC:entire atmosphere:7 hour fcst:
637:436813175:d=2026111100:TCDC:entire atmosphere:6-7 hour ave fcst:
653:451448385:d=2026111100:DSWRF:surface:6-7 hour ave fcst:
654:452000000:d=2026111100:UNKNOWN:surface:7 hour fcst:
"""

F024 = """\
596:429749287:d=2026111100:APCP:surface:18-24 hour acc fcst:
597:430105470:d=2026111100:APCP:surface:0-1 day acc fcst:
598:430500000:d=2026111100:ACPCP:surface:0-1 day acc fcst:
"""


def test_forecast_hours_match_gfs_output_schedule():
    hours = forecast_hours()
    assert len(hours) == 209
    assert hours[:3] == [0, 1, 2] and hours[120] == 120 and hours[121] == 123 and hours[-1] == 384
    assert forecast_hours(24) == list(range(25))


def test_parse_idx_computes_inclusive_byte_ranges():
    recs = parse_idx(F007)
    assert recs[0].param == "TMP" and recs[0].start == 407306589 and recs[0].end == 408500000 - 1
    assert recs[-1].end is None


def test_parse_fcst_variants():
    assert parse_fcst("anl") == FcstSpec("instant", None, 0)
    assert parse_fcst("7 hour fcst") == FcstSpec("instant", None, 7)
    assert parse_fcst("6-7 hour acc fcst") == FcstSpec("acc", 6, 7)
    assert parse_fcst("0-1 day acc fcst") == FcstSpec("acc", 0, 24)
    assert parse_fcst("120-126 hour ave fcst") == FcstSpec("ave", 120, 126)
    assert parse_fcst("something else") is None


def test_select_record_picks_the_right_message():
    recs = parse_idx(F007)
    rec, spec = select_record(recs, "TCDC", "entire atmosphere", "instant", 7)
    assert rec.fcst == "7 hour fcst"
    rec, spec = select_record(recs, "APCP", "surface", "accum", 7)
    assert rec.fcst == "0-7 hour acc fcst" and spec.start == 0
    rec, spec = select_record(recs, "DSWRF", "surface", "average", 7)
    assert spec == FcstSpec("ave", 6, 7)
    assert select_record(recs, "TMP", "2 m above ground", "instant", 6) is None


def test_select_record_handles_day_units():
    rec, spec = select_record(parse_idx(F024), "APCP", "surface", "accum", 24)
    assert rec.fcst == "0-1 day acc fcst"


def test_merge_ranges_coalesces_nearby_messages():
    recs = parse_idx(F007)
    tmp, _, dpt = recs[0], recs[1], recs[2]
    groups = merge_ranges([tmp, dpt], max_gap=2_000_000)
    assert len(groups) == 1 and groups[0].start == tmp.start and groups[0].end == dpt.end
    assert len(merge_ranges([tmp, dpt], max_gap=0)) == 2
