for machine in "valve" "fan" "slider" "pump"
do
        for id in "00" "02" "04" "06"
        do
                echo "${machine} - ${id}"
                python nearest_neighbour.py -d ../data/0db/${machine}/id_${id}
        done
done