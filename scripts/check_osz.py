import pathlib, zipfile
osz = list(pathlib.Path("data/osz").glob("*.osz"))
ok = bad = no_osu = 0
for p in osz:
    if not zipfile.is_zipfile(p):
        bad += 1; print("깨진 zip:", p.name); continue
    with zipfile.ZipFile(p) as z:
        if not any(n.endswith(".osu") for n in z.namelist()):
            no_osu += 1; print(".osu 없음:", p.name); continue
    ok += 1
print(f"정상 {ok} / 깨짐 {bad} / .osu없음 {no_osu} (총 {len(osz)})")