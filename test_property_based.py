from hypothesis.extra.pandas import data_frames
import pandas as pd
import hypothesis.strategies as st
from hypothesis import given

# Function to test


def add_numbers(a, b):
    return a + b

# Property-based test


@given(st.integers(), st.integers())
def test_add_numbers(a, b):
    result = add_numbers(a, b)

    # Check the property: the sum of the two numbers should be equal to the result
    assert result == a + b


# Function to test

def sort_list(lst):
    return sorted(lst)

# Property-based test


@given(st.lists(st.integers()))
def test_sort_list(lst):
    sorted_lst = sort_list(lst)

    # Check the property: the sorted list should be in ascending order
    assert sorted_lst == sorted(lst)
    assert sorted_lst == sorted(lst, reverse=True)[::-1]


# Function to test

def temporal_train_test_split(data, split_date):
    train = data.query("date < @split_date")
    test = data.query("date > @split_date")

    return train, test

# Property-based test


@given(
    data=data_frames(
        index=st.integers(min_value=0, max_value=100),
        columns=[
            st.column("date", dtype=str),
            st.column("x1", dtype=int),
            st.column("x2", dtype=str),
        ],
    ),
    split_date=st.dates().map(lambda d: d.strftime("%Y-%m-%d")),
)
def test_temporal_train_test_split(data, split_date):
    train, test = temporal_train_test_split(data, split_date)

    # Property 1: All dates in train should be less than or equal to split_date
    assert (train["date"] <= split_date).all()

    # Property 2: All dates in test should be greater than split_date
    assert (test["date"] > split_date).all()

    # Property 3: Concatenating train and test should result in the original data frame
    concatenated = pd.concat([train, test]).sort_values(
        ["date", "x1", "x2"]).reset_index(drop=True)
    sorted_input = data.sort_values(
        ["date", "x1", "x2"]).reset_index(drop=True)
    assert concatenated.equals(sorted_input)
