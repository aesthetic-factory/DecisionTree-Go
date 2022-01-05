package main

import "fmt"

func main() {

	var ds DataSet
	if ds.loadDataFromFile("./testing_data/data.csv") == nil {
		var dt DecisionTree
		dt.train(ds)
		// dt.show()
		for i := 0; i < 6; i++ {
			err, res := dt.inference(ds[i].attributes)
			if err == nil {
				fmt.Println(i, " ", res)
			}
		}
	}
}
