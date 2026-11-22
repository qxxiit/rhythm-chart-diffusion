#!/bin/sh
# 11/23 (~20 min): the main playtest's pack (11/30-12/6). 16 test songs, one chart each, the
# human chart or the system's (full-v4 through scripts/system.sh as decided, DECISIONS
# 2026-11-23), half and half, in two versions (the second has each song's other chart; one
# version per player, half the players each). The songs: at most 2 min long, SR 1.5-4.5, long
# notes at most 30% of the human chart's onsets (charts heavier in long notes were skipped as
# too hard on 11/23), spread over SR; only those 16 are drawn. answers.csv is the key: keep it
# from the players and from anyone who will play. Why: EXPERIMENTS 2026-11-23 (2). Writes
# under outputs/ only, nothing is deleted.
#   songs   (seconds) the 16 songs' chart keys -> outputs/playtest_main/songs.txt
#   system  (~15 min) the system on them (OUT=outputs/system_playtest)
#   pack    (~1 min) outputs/playtest_main/playtest_pack_v1.zip and _v2.zip
#   check   (seconds) counts only, nothing of the key
# All: for p in songs system pack check; do sh experiments/runs/run_1140.sh $p; done
set -ex
PHASE=${1:-songs}
P=outputs/playtest_main
S=outputs/system_playtest
TAG=${TAG:-1130}
PICK="--single --split test --songs 16 --sr-range 1.5 4.5 --max-seconds 120 --max-hold-share 0.3"

case $PHASE in
songs)
    [ -f $P/songs.txt ] && { echo "$P/songs.txt is there"; exit 0; }
    python scripts/playtest_pack.py $PICK --list-songs $P/songs.txt
    ;;
system)
    [ -f $P/songs.txt ] || { echo "no $P/songs.txt"; exit 2; }
    [ -f $S/test_n0.path ] && { echo "the system's charts are there"; exit 0; }
    OUT=$S SONGS=$P/songs.txt sh scripts/system.sh test 0
    ;;
pack)
    [ -f $S/test_n0.path ] || { echo "no $S/test_n0.path"; exit 2; }
    [ -f $P/answers.csv ] && { echo "the pack is there"; exit 0; }
    python scripts/playtest_pack.py --single --split test --from-run "$(cat $S/test_n0.path)" \
        --keys $(cat $P/songs.txt) --versions 2 --tag $TAG --out $P
    ;;
check)
    for v in 1 2; do
        echo "pack_v$v: $(ls $P/pack_v$v/*.osz | wc -l) .osz, $(($(wc -l < $P/pack_v$v/ratings.csv) - 1)) rating rows"
        tail -n 1 $P/pack_v$v/README.txt
        ls -l $P/playtest_pack_v$v.zip
    done
    echo "songs listed: $(wc -l < $P/songs.txt); key rows: $(($(wc -l < $P/answers.csv) - 1))"
    ;;
*)
    echo "phases: songs system pack check"; exit 2
    ;;
esac
