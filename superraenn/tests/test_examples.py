from ..feature_extraction import read_in_LC_files

def test_load_LC():
	my_message = read_in_LC_files(['blah'])
	print(my_message)
	assert 1==1