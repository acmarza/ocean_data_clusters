import argparse
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True,
                        help="netCDF (.nc) file to modify")
    args = parser.parse_args()

    data_arr = xr.load_dataarray(args.file)
    print(f"current long_name: {data_arr.attrs}")
    new_name = input("[>] enter new long_name: ")
    data_arr.attrs['long_name'] = new_name
    print(f"current long_name: {data_arr.attrs}")
    ok = input("save to file? [y/n]: ")
    data_arr.close()
    if ok == 'y':
        data_arr.to_netcdf(args.file)
    print("[i] saved")


if __name__ == "__main__":
    main()
