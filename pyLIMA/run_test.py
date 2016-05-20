from manager import main


class FakeArguments(object):
    pass


def test_something():
    arguments = FakeArguments()
    arguments.input_directory = '/home/mnorbury/Microlensing/Space/Lightcurves/'
    arguments.output_directory = '/home/mnorbury/Microlensing/'
    arguments.claret = '/home/mnorbury/Microlensing/Claret2011/J_A+A_529_A75/'
    arguments.model = 'PSPL'

    # path = arguments.input_directory + 'FSPL' + '/Lightcurves/'

    main(arguments)

    assert 1 == 1
