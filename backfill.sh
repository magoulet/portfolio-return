startdate=2022-02-14
n=$startdate
enddate=2022-04-04
i=0


while [ "$n" != "$enddate" ]; do
	n=$(date +%Y-%m-%d -d "$startdate + $i days")
	python3 Portfolio_Return.py -d $n
	echo $n
	((i++))
done
