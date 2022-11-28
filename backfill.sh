startdate=2022-06-15
n=$startdate
enddate=2022-07-20
i=0


while [ "$n" != "$enddate" ]; do
	n=$(date +%Y-%m-%d -d "$startdate + $i days")
	python3 Portfolio_Return_oo.py -d $n
	echo $n
	((i++))
done
