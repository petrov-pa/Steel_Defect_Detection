project\flask_run.py:0:1: DC100 Module has no docstring
project\flask_run.py:1:1: D100 Missing docstring in public module
project\flask_run.py:15:1: DC102 Missing docstring in public function run
project\flask_run.py:15:1: D103 Missing docstring in public function
project\flask_run.py:15:9: ANN001 Missing type annotation for function argument 'filename'
project\flask_run.py:16:0: ANN201 Missing return type annotation for public function
project\flask_run.py:28:5: ECE001 Expression is too complex (8.0 > 7)
project\flask_run.py:32:9: ECE001 Expression is too complex (8.0 > 7)
project\flask_run.py:54:5: C408 Unnecessary dict call - rewrite as a literal.
project\flask_run.py:60:1: DC102 Missing docstring in public function bad_request
project\flask_run.py:60:1: D103 Missing docstring in public function
project\flask_run.py:60:1: CF001 Avoid `route` decorator. Use a suitable HTTP method as decorator.
project\flask_run.py:60:18: ANN201 Missing return type annotation for public function
project\flask_run.py:64:1: DC101 Missing docstring in class MyForm
project\flask_run.py:64:1: D101 Missing docstring in public class
project\flask_run.py:65:5: VNE002 variable name 'file' should be clarified
project\flask_run.py:75:1: DC102 Missing docstring in public function predict
project\flask_run.py:75:1: D103 Missing docstring in public function
project\flask_run.py:75:1: CF001 Avoid `route` decorator. Use a suitable HTTP method as decorator.
project\flask_run.py:75:14: ANN201 Missing return type annotation for public function
project\flask_run.py:78:9: VNE002 variable name 'file' should be clarified


project\main.py:0:1: DC100 Module has no docstring
project\main.py:1:1: D100 Missing docstring in public module
project\main.py:9:1: DC102 Missing docstring in public function main
project\main.py:9:1: D103 Missing docstring in public function
project\main.py:9:11: ANN201 Missing return type annotation for public function
project\main.py:27:43: E203 whitespace before ':'

project\src\load_test_data.py:0:1: DC100 Module has no docstring
project\src\load_test_data.py:1:1: D100 Missing docstring in public module
project\src\load_test_data.py:6:89: E501 line too long (92 > 88 characters)

project\src\load_train_data.py:0:1: DC100 Module has no docstring
project\src\load_train_data.py:1:1: D100 Missing docstring in public module

project\src\load_weights.py:0:1: DC100 Module has no docstring
project\src\load_weights.py:1:1: D100 Missing docstring in public module

project\src\models.py:14:1: E302 expected 2 blank lines, found 1
project\src\models.py:15:1: D205 1 blank line required between summary line and description
project\src\models.py:15:1: D209 Multi-line docstring closing quotes should be on a separate line
project\src\models.py:15:1: D400 First line should end with a period
project\src\models.py:15:1: D403 First word of the first line should be properly capitalized
project\src\models.py:33:12: R504 unnecessary variable assignment before return statement.
project\src\models.py:36:18: ANN201 Missing return type annotation for public function
project\src\models.py:37:1: D210 No whitespaces allowed surrounding docstring text
project\src\models.py:37:1: D400 First line should end with a period
project\src\models.py:37:1: D403 First word of the first line should be properly capitalized
project\src\models.py:44:12: R504 unnecessary variable assignment before return statement.
project\src\models.py:47:1: DC101 Missing docstring in class FixedDropout
project\src\models.py:47:1: D101 Missing docstring in public class
project\src\models.py:48:5: DC103 Missing docstring in private function _get_noise_shape
project\src\models.py:48:26: ANN101 Missing type annotation for self in method
project\src\models.py:48:32: ANN001 Missing type annotation for function argument 'inputs'
project\src\models.py:48:39: ANN202 Missing return type annotation for protected function

project\src\train_classifier.py:0:1: DC100 Module has no docstring
project\src\train_classifier.py:1:1: D100 Missing docstring in public module
project\src\train_classifier.py:9:20: PD901 'df' is a bad variable name. Be kinder to your future self.

project\src\train_linknet.py:0:1: DC100 Module has no docstring
project\src\train_linknet.py:1:1: D100 Missing docstring in public module
project\src\train_linknet.py:10:20: PD901 'df' is a bad variable name. Be kinder to your future self.

project\src\metrics\losses.py:0:1: DC100 Module has no docstring
project\src\metrics\losses.py:1:1: D100 Missing docstring in public module
project\src\metrics\losses.py:7:15: ANN001 Missing type annotation for function argument 'true'
project\src\metrics\losses.py:7:21: ANN001 Missing type annotation for function argument 'pred'
project\src\metrics\losses.py:7:26: ANN201 Missing return type annotation for public function
project\src\metrics\losses.py:8:1: D205 1 blank line required between summary line and description
project\src\metrics\losses.py:8:1: D208 Docstring is over-indented
project\src\metrics\losses.py:17:19: ANN001 Missing type annotation for function argument 'true'
project\src\metrics\losses.py:17:25: ANN001 Missing type annotation for function argument 'pred'
project\src\metrics\losses.py:17:30: ANN201 Missing return type annotation for public function
project\src\metrics\losses.py:18:1: D205 1 blank line required between summary line and description
project\src\metrics\losses.py:18:1: D208 Docstring is over-indented
project\src\metrics\losses.py:18:1: D401 First line should be in imperative mood
project\src\metrics\losses.py:26:1: DC102 Missing docstring in public function weighted_loss
project\src\metrics\losses.py:26:1: D103 Missing docstring in public function
project\src\metrics\losses.py:26:19: ANN001 Missing type annotation for function argument 'original_loss_func'
project\src\metrics\losses.py:26:39: ANN001 Missing type annotation for function argument 'weights_list'
project\src\metrics\losses.py:26:52: ANN201 Missing return type annotation for public function
project\src\metrics\losses.py:27:5: DC102 Missing docstring in public function loss_func
project\src\metrics\losses.py:27:19: ANN001 Missing type annotation for function argument 'true'
project\src\metrics\losses.py:27:25: ANN001 Missing type annotation for function argument 'pred'
project\src\metrics\losses.py:28:0: ANN201 Missing return type annotation for public function
project\src\metrics\losses.py:42:16: R504 unnecessary variable assignment before return statement.

project\src\preprocessing\data_generator.py:0:1: DC100 Module has no docstring
project\src\preprocessing\data_generator.py:1:1: D100 Missing docstring in public module
project\src\preprocessing\data_generator.py:8:1: DC101 Missing docstring in class DataGenerator
project\src\preprocessing\data_generator.py:8:1: D101 Missing docstring in public class
project\src\preprocessing\data_generator.py:9:1: D107 Missing docstring in __init__
project\src\preprocessing\data_generator.py:9:5: DC104 Missing docstring in special function __init__
project\src\preprocessing\data_generator.py:9:5: CFQ002 Function "__init__" has 7 arguments that exceeds max allowed 6
project\src\preprocessing\data_generator.py:10:15: ANN001 Missing type annotation for function argument 'img_name'
project\src\preprocessing\data_generator.py:10:25: ANN001 Missing type annotation for function argument 'img_data'
project\src\preprocessing\data_generator.py:10:35: ANN001 Missing type annotation for function argument 'batch_size'
project\src\preprocessing\data_generator.py:10:49: ANN001 Missing type annotation for function argument 'shuffle'
project\src\preprocessing\data_generator.py:10:63: ANN001 Missing type annotation for function argument 'aug'
project\src\preprocessing\data_generator.py:10:73: ANN001 Missing type annotation for function argument 'seg'
project\src\preprocessing\data_generator.py:11:6: ANN204 Missing return type annotation for special method
project\src\preprocessing\data_generator.py:21:1: D105 Missing docstring in magic method
project\src\preprocessing\data_generator.py:21:5: DC104 Missing docstring in special function __len__
project\src\preprocessing\data_generator.py:22:0: ANN204 Missing return type annotation for special method
project\src\preprocessing\data_generator.py:25:1: D105 Missing docstring in magic method
project\src\preprocessing\data_generator.py:25:5: DC104 Missing docstring in special function __getitem__
project\src\preprocessing\data_generator.py:25:27: ANN001 Missing type annotation for function argument 'index'
project\src\preprocessing\data_generator.py:26:0: ANN204 Missing return type annotation for special method
project\src\preprocessing\data_generator.py:28:36: E203 whitespace before ':'
project\src\preprocessing\data_generator.py:31:10: N806 variable 'X' in function should be lowercase
project\src\preprocessing\data_generator.py:34:1: D102 Missing docstring in public method
project\src\preprocessing\data_generator.py:34:5: DC102 Missing docstring in public function on_epoch_end
project\src\preprocessing\data_generator.py:35:0: ANN201 Missing return type annotation for public function
project\src\preprocessing\data_generator.py:40:1: D102 Missing docstring in public method
project\src\preprocessing\data_generator.py:40:5: DC102 Missing docstring in public function augmentations
project\src\preprocessing\data_generator.py:40:29: ANN001 Missing type annotation for function argument 'image'
project\src\preprocessing\data_generator.py:40:36: ANN001 Missing type annotation for function argument 'label'
project\src\preprocessing\data_generator.py:41:0: ANN201 Missing return type annotation for public function
project\src\preprocessing\data_generator.py:48:5: DC103 Missing docstring in private function __data_generation
project\src\preprocessing\data_generator.py:48:33: ANN001 Missing type annotation for function argument 'list_id_temp'
project\src\preprocessing\data_generator.py:48:46: ANN203 Missing return type annotation for secret function
project\src\preprocessing\data_generator.py:49:22: C408 Unnecessary list call - rewrite as a literal.
project\src\preprocessing\data_generator.py:50:24: C408 Unnecessary list call - rewrite as a literal.
project\src\preprocessing\data_generator.py:59:45: E203 whitespace before ':'
project\src\preprocessing\data_generator.py:60:49: E203 whitespace before ':'
project\src\preprocessing\data_generator.py:64:93: LN002 doc/comment line is too long (98 > 92)
project\src\preprocessing\data_generator.py:64:93: E501 line too long (98 > 92 characters)

project\src\preprocessing\preprocessing.py:0:1: DC100 Module has no docstring
project\src\preprocessing\preprocessing.py:1:1: D100 Missing docstring in public module
project\src\preprocessing\preprocessing.py:5:19: ANN201 Missing return type annotation for public function
project\src\preprocessing\preprocessing.py:6:1: D210 No whitespaces allowed surrounding docstring text
project\src\preprocessing\preprocessing.py:12:1: E800 Found commented out code
project\src\preprocessing\preprocessing.py:16:12: R504 unnecessary variable assignment before return statement.
project\src\preprocessing\preprocessing.py:19:14: ANN001 Missing type annotation for function argument 'name'
project\src\preprocessing\preprocessing.py:19:20: ANN001 Missing type annotation for function argument 'df'
project\src\preprocessing\preprocessing.py:19:23: ANN201 Missing return type annotation for public function
project\src\preprocessing\preprocessing.py:20:1: D205 1 blank line required between summary line and description
project\src\preprocessing\preprocessing.py:20:1: D401 First line should be in imperative mood
project\src\preprocessing\preprocessing.py:28:29: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\preprocessing.py:39:30: E203 whitespace before ':'

project\src\preprocessing\split_data.py:0:1: DC100 Module has no docstring
project\src\preprocessing\split_data.py:1:1: D100 Missing docstring in public module
project\src\preprocessing\split_data.py:7:22: ANN201 Missing return type annotation for public function
project\src\preprocessing\split_data.py:8:1: D205 1 blank line required between summary line and description
project\src\preprocessing\split_data.py:8:1: D208 Docstring is over-indented
project\src\preprocessing\split_data.py:8:1: D210 No whitespaces allowed surrounding docstring text
project\src\preprocessing\split_data.py:18:5: PD901 'df' is a bad variable name. Be kinder to your future self.
project\src\preprocessing\split_data.py:19:95: LN002 doc/comment line is too long (95 > 94)
project\src\preprocessing\split_data.py:19:95: E501 line too long (95 > 94 characters)
project\src\preprocessing\split_data.py:21:24: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:23:13: PD901 'df' is a bad variable name. Be kinder to your future self.
project\src\preprocessing\split_data.py:31:9: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:37:21: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:38:21: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:39:21: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:40:21: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:44:5: PD002 'inplace = True' should be avoided; it has inconsistent behavior
project\src\preprocessing\split_data.py:48:17: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:52:21: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
project\src\preprocessing\split_data.py:54:5: ECE001 Expression is too complex (7.5 > 7)
project\src\preprocessing\split_data.py:54:17: PD011 Use '.array' or '.to_array()' instead of '.values'; 'values' is ambiguous
