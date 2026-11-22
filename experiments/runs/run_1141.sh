#!/bin/sh
# 11/23 (~15-30 min): AI charts of songs the user picked (arcade songs, not in the data), to
# play: the system as decided (scripts/generate_system.py: the draw chosen by sel_holds, then
# the passes), SR 2.5, 3.5 and 4.5 each, long notes at most 30%. Not an experiment and no
# human charts: the user plays them and writes what felt right or wrong. Put the songs'
# beatmap sets (.osz, any mode: only their audio and red lines are used) in inputs/arcade/
# (gitignored: the audio stays off the repository). Writes under outputs/ only.
#   charts  (~3-5 min a song, three at a time) outputs/arcade_1123/<song>/<song>.osz
#   pack    (seconds) outputs/arcade_1123/arcade_1123.zip: the .osz files, README, a sheet
# All: for p in charts pack; do sh experiments/runs/run_1141.sh $p; done
set -ex
PHASE=${1:-charts}
IN=${IN:-inputs/arcade}
O=outputs/arcade_1123
SR=${SR:-2.5 3.5 4.5}
JOBS=${JOBS:-3}

case $PHASE in
charts)
    ls "$IN"/*.osz > /dev/null || { echo "no .osz in $IN"; exit 2; }
    mkdir -p $O
    n=0
    for f in "$IN"/*.osz; do
        stem=$(basename "$f" .osz)
        [ -f "$O/$stem/$stem.osz" ] && { echo "$stem is there"; continue; }
        python scripts/generate_system.py --osz "$f" --sr $SR --out "$O/$stem" \
            > "$O/$stem.log" 2>&1 &
        n=$((n + 1))
        [ $((n % JOBS)) -eq 0 ] && wait
    done
    wait
    for f in "$IN"/*.osz; do
        stem=$(basename "$f" .osz)
        [ -f "$O/$stem/$stem.osz" ] || { echo "no chart for $stem: see $O/$stem.log"; tail -n 5 "$O/$stem.log"; }
    done
    ;;
pack)
    mkdir -p $O/pack
    for d in $O/*/; do
        [ "$d" = "$O/pack/" ] && continue
        stem=$(basename "$d")
        [ -f "$d$stem.osz" ] && cp "$d$stem.osz" $O/pack/
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
