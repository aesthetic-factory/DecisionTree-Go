package main

import (
	"errors"
	"fmt"
	"sync"
)

const MIN_SAMPLE = 50
const MIN_GINI_GAIN = 0.0001

var thread_in_use = 0

type Condition struct {
	column uint16
	value  int32
}

func (condition *Condition) match(attributes []int32) bool {
	if attributes[condition.column] <= condition.value {
		return true
	} else {
		return false
	}
}

type TreeNode struct {
	isLeaf    bool
	leafValue map[int32]int32

	condition Condition

	leftNode  *TreeNode
	rightNode *TreeNode
}

type DecisionTree struct {
	node          TreeNode
	attribute_num int
}

func (dt *DecisionTree) show() {
	dt.node.show()
}

func (dt *DecisionTree) train(ds DataSet) error {
	if len(ds) == 0 {
		return errors.New("DataSet is empty")
	}
	dt.attribute_num = len(ds[0].attributes)
	dt.node = TreeNode{}
	return dt.node.train(ds)
}

func (dt *DecisionTree) inference(attributes []int32) (error, map[int32]int32) {
	return dt.node.inference(attributes)
}
func (dt *DecisionTree) save() bool {
	return true
}

func (dt *DecisionTree) load() bool {
	return true
}

func (node *TreeNode) show() error {
	if node.isLeaf {
		fmt.Println("condition: ", node.condition, "node.isLeaf: ", node.isLeaf, " node.leafValue: ", node.leafValue)
	} else {
		fmt.Println("condition: ", node.condition, "node.isLeaf: ", node.isLeaf)
	}
	if node.leftNode != nil {
		node.leftNode.show()
	}
	if node.rightNode != nil {
		node.rightNode.show()
	}
	return nil
}

func (node *TreeNode) train(ds DataSet) error {
	if len(ds) == 0 {
		return errors.New("DataSet is empty")
	}
	gain, condition := find_best_split(ds, MIN_SAMPLE)
	left_ds, right_ds := partition(ds, condition)
	node.condition = condition

	if gain < MIN_GINI_GAIN || len(ds) < MIN_SAMPLE {
		label_counts := countLabel(&ds)
		node.isLeaf = true
		node.leafValue = label_counts
		return nil
	}

	node.leftNode = new(TreeNode)
	node.rightNode = new(TreeNode)

	node.trainMT(node.leftNode, node.rightNode, left_ds, right_ds)
	return nil
}

func (node *TreeNode) trainMT(leftNode *TreeNode, rightNode *TreeNode, left_ds DataSet, right_ds DataSet) error {
	var allowThread = false
	if thread_in_use < 20 && len(left_ds) > MIN_SAMPLE*3 {
		allowThread = true
	} else if thread_in_use < 40 && len(left_ds) > MIN_SAMPLE*10 {
		allowThread = true
	} else if thread_in_use < 60 && len(left_ds) > MIN_SAMPLE*20 {
		allowThread = true
	}

	if allowThread {
		var wg sync.WaitGroup
		wg.Add(1)
		thread_in_use++
		go func() {
			leftNode.train(left_ds)
			wg.Done()
			thread_in_use--
		}()
		rightNode.train(right_ds)
		wg.Wait()
	} else {
		leftNode.train(left_ds)
		rightNode.train(right_ds)
	}
	return nil
}

func (node *TreeNode) inference(attributes []int32) (error, map[int32]int32) {
	if node.isLeaf {
		return nil, node.leafValue
	}
	if node.condition.match(attributes) {
		// left node
		return node.leftNode.inference(attributes)
	} else {
		// right node
		return node.rightNode.inference(attributes)
	}
}

func find_best_split(ds DataSet, minSample int) (float64, Condition) {
	/* Find the best question to ask by iterating over every feature / value
	   and calculating the information gain.
	*/
	best_gain := float64(0) // keep track of the best information gain

	// keep train of the feature / value that produced it
	best_condition := Condition{column: 0, value: 0}
	current_uncertainty := gini(ds)
	n_features := uint16(len(ds[0].attributes)) // number of columns

	for col := uint16(0); col < n_features; col++ { // for each feature
		values := countAttributes(&ds, col)
		for val := range values { // for each value

			condition := Condition{column: col, value: val}

			// try splitting the dataset
			true_records, false_records := partition(ds, condition)

			// Skip this split if it doesn't divide the
			// dataset.
			if len(true_records) < minSample || len(false_records) < minSample {
				continue
			}

			// Calculate the information gain from this split
			gain := gini_gain(true_records, false_records, current_uncertainty)

			if gain > best_gain {
				best_gain, best_condition = gain, condition
			}
		}
	}
	return best_gain, best_condition
}

func gini_gain(left DataSet, right DataSet, current_gini float64) float64 {
	var p = float64(len(left)) / float64(len(left)+len(right))
	return current_gini - p*gini(left) - (1-p)*gini(right)
}

func gini(ds DataSet) float64 {
	impurity := float64(1)
	label_counts := countLabel(&ds)

	for _, label := range label_counts {
		prob_of_lbl := float64(label) / float64(len(ds))
		impurity -= prob_of_lbl * (1.0 - prob_of_lbl)
	}
	return 1 - impurity
}

func partition(ds DataSet, condition Condition) (DataSet, DataSet) {
	true_records := DataSet{}
	false_records := DataSet{}
	for _, record := range ds {
		if condition.match(record.attributes) {
			true_records = append(true_records, record)
		} else {
			false_records = append(false_records, record)
		}
	}
	return true_records, false_records
}
