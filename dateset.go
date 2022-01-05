package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

type TrainingRecord struct {
	label      int32
	attributes []int32
}

type DataSet []TrainingRecord

func (ds *DataSet) loadDataFromFile(file string) error {
	pFile, err := os.Open(file)
	if err != nil {
		fmt.Println(err)
		return err
	}

	fmt.Println("Successfully Opened CSV file")
	defer pFile.Close()

	csvLines, err := csv.NewReader(pFile).ReadAll()
	if err != nil {
		fmt.Println(err)
		return err
	}

	for row, line := range csvLines {
		if row == 0 {
			continue // skip header
		}

		var record TrainingRecord
		var num_attribute = len(line) - 1 // -1 to skip first column, it is label
		record.attributes = make([]int32, num_attribute)

		// for every columns in a line
		for col, val_str := range line {
			if val, err := strconv.ParseInt(val_str, 10, 32); err != nil {
				fmt.Printf("Unable to parse %s on line %d", line[col], row)
			} else {
				if col == 0 {
					record.label = int32(val) // first column is label
				} else {
					record.attributes[col-1] = int32(val)
					// record.attributes = append(record.attributes, int32(val)) // the rests are attributes
				}
			}
		}
		*ds = append(*ds, record)
	}
	return nil
}

func countLabel(ds *DataSet) map[int32]int32 {
	set := make(map[int32]int32)
	for _, record := range *ds {
		_, ok := set[record.label]
		if ok {
			set[record.label] += 1
		} else {
			set[record.label] = 1
		}
	}
	return set
}

func countAttributes(ds *DataSet, column uint16) map[int32]int32 {
	set := make(map[int32]int32)
	for _, record := range *ds {
		_, ok := set[record.attributes[column]]
		if ok {
			set[record.attributes[column]] += 1
		} else {
			set[record.attributes[column]] = 1
		}
	}
	return set
}

func countUniqueLabel(ds *DataSet) int {
	set := make(map[int32]struct{})
	for _, record := range *ds {
		set[record.label] = struct{}{} // assign empty struct as value
	}
	return len(set)
}

func countUniqueAttributes(ds *DataSet, column uint16) int {
	set := make(map[int32]struct{})
	for _, record := range *ds {
		set[record.attributes[column]] = struct{}{} // assign empty struct as value
	}
	return len(set)
}
