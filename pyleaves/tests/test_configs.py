
from pyleaves.utils import config_utils

def test_get_config_uuid():
    a={'name':'Danish', 'age':107}
    b={'age':107, 'name':'Danish'}

    assert get_config_uuid(a) == get_config_uuid(b)
