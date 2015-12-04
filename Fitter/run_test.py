from manager import main


class FakeArguments(object):
    pass


def test_something():
    arguments = FakeArguments()
    arguments.input_directory = '/home/mnorbury/Microlensing/'
    arguments.output_directory = '/home/mnorbury/Microlensing/'

    path = arguments.input_directory + 'FSPL' + '/Lightcurves/'

    main(path, arguments)

    assert 1 == 1
