#!/bin/sh
# 11/23 (~15-30 min): AI charts of songs the user picked (arcade songs, not in the data), to
# play: the system as decided (scripts/generate_system.py: the draw chosen by sel_holds, then
# the passes), SR 2.5, 3.5 and 4.5 each, long notes at most 30%. Not an experiment and no
# human charts: the user plays them and writes what felt right or wrong. The songs come as
# beatmap sets (.osz, any mode: only their audio and red lines are used) in inputs/arcade/
# (gitignored: the audio stays off the repository), fetched by download from the beatmap ids
# in inputs/arcade/ids.txt (osu.ppy.sh/b/ID names the set; the mirrors data/ was downloaded
# from give it). Writes under inputs/arcade/ and outputs/ only.
#   download (~1 min) inputs/arcade/<beatmap id>_<set id>.osz, the songs' titles printed
#   charts  (~3-5 min a song, three at a time) outputs/arcade_1123/<id>/<artist - title>.osz
#   pack    (seconds) outputs/arcade_1123/arcade_1123.zip: the .osz files, README, a sheet
# All: for p in download charts pack; do sh experiments/runs/run_1141.sh $p; done
set -ex
PHASE=${1:-charts}
IN=${IN:-inputs/arcade}
O=outputs/arcade_1123
SR=${SR:-2.5 3.5 4.5}
JOBS=${JOBS:-3}

case $PHASE in
download)
    [ -f "$IN/ids.txt" ] || { echo "no $IN/ids.txt (beatmap ids, one a line)"; exit 2; }
    UA="rhythm-chart-diffusion (osu!mania research)"
    for bid in $(cat "$IN/ids.txt"); do
        ls "$IN/${bid}"_*.osz > /dev/null 2>&1 && { echo "$bid is there"; continue; }
        url=$(curl -sL -A "$UA" -o /dev/null -w '%{url_effective}' "https://osu.ppy.sh/b/$bid" || true)
        sid=$(echo "$url" | sed -n 's#.*/beatmapsets/\([0-9][0-9]*\).*#\1#p')
        [ -n "$sid" ] || sid=$(curl -s -A "$UA" "https://catboy.best/api/v2/b/$bid" 2>/dev/null \
            | sed -n 's/.*"beatmapset_id": *\([0-9][0-9]*\).*/\1/p' | head -n 1 || true)
        [ -n "$sid" ] || { echo "beatmap $bid: no set found ($url)"; continue; }
        f="$IN/${bid}_$sid.osz"
        for m in "https://catboy.best/d/$sid" "https://api.nerinyan.moe/d/$sid"; do
            curl -sL --fail -m 300 -A "$UA" -o "$f.part" "$m" \
                && [ "$(wc -c < "$f.part")" -gt 10000 ] && mv "$f.part" "$f" && break
        done
        if [ -f "$f" ]; then
            echo "beatmap $bid -> set $sid: $(unzip -p "$f" '*.osu' 2>/dev/null \
                | sed -n 's/^Title:\(.*\)/\1/p' | head -n 1 | tr -d '\r')"
        else
            echo "beatmap $bid -> set $sid: the download failed"
        fi
        sleep 2
    done
    ;;
charts)
    ls "$IN"/*.osz > /dev/null || { echo "no .osz in $IN"; exit 2; }
    mkdir -p $O
    n=0
    for f in "$IN"/*.osz; do
        stem=$(basename "$f" .osz)
        ls "$O/$stem"/*.osz > /dev/null 2>&1 && { echo "$stem is there"; continue; }
        python scripts/generate_system.py --osz "$f" --sr $SR --out "$O/$stem" \
            > "$O/$stem.log" 2>&1 &
        n=$((n + 1))
        [ $((n % JOBS)) -eq 0 ] && wait
    done
    wait
    for f in "$IN"/*.osz; do
        stem=$(basename "$f" .osz)
        ls "$O/$stem"/*.osz > /dev/null 2>&1 \
            || { echo "no chart for $stem: see $O/$stem.log"; tail -n 5 "$O/$stem.log"; }
    done
    ;;
pack)
    mkdir -p $O/pack
    for d in $O/*/; do
        [ "$d" = "$O/pack/" ] && continue
        for f in "$d"*.osz; do [ -f "$f" ] && cp "$f" $O/pack/; done
    done
    cat > $O/pack/README.txt <<'EOF'
AI 채보 플레이 (rhythm-chart-diffusion, 11/23)

곡마다 AI 가 만든 채보 셋 (AI 2.5 / AI 3.5 / AI 4.5: 목표 난이도 SR).
지금 시스템 그대로이고, 사람 채보는 없습니다.

1. .osz 를 osu!(lazer) 에 끌어다 놓거나 더블클릭 (stable: Songs 폴더 + F5).
2. 치고 싶은 난이도를 쳐 보고, 피드백.txt 에 곡마다 한 줄: 좋았던 곳, 어색했던 곳
   (몇 초쯤인지 적어 주면 더 좋음).
EOF
    {
        echo "곡 | 난이도 | 좋았던 곳 | 어색했던 곳"
        for f in $O/pack/*.osz; do echo "$(basename "$f" .osz) | | |"; done
    } > $O/pack/피드백.txt
    [ -f $O/arcade_1123.zip ] && { echo "$O/arcade_1123.zip is there"; exit 0; }
    (cd $O/pack && zip -q -r ../arcade_1123.zip .)
    ls -l $O/arcade_1123.zip
    ;;
*)
    echo "phases: charts pack"; exit 2
    ;;
esac
