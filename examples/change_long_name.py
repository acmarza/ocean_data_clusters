import argparse
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True,
                        help="netCDF (.nc) file to modify")
    parser.add_argument('-n', '--name', required=False,
                        help='(optional) provide new long_name on commandline')
    args = parser.parse_args()

    data_arr = xr.load_dataarray(args.file)
    print(f"current long_name: {data_arr.attrs}")
    if args.name is None:
        new_name = input("[>] enter new long_name: ")
    else:
        new_name = args.name
    data_arr.attrs['long_name'] = new_name
    print(f"new long_name: {data_arr.attrs}")
    data_arr.close()
    data_arr.to_netcdf(args.file)
    print("[i] saved")


if __name__ == "__main__":
    main()
