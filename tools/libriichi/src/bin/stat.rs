use std::env;

use anyhow::{Context, Result};
use glob::glob;
use riichi::stat::Stat;

const USAGE: &str = "Usage: stat <DIR> <PLAYER_NAME>";

fn main() -> Result<()> {


    // let args: Vec<_> = env::args().collect();
    let s1 = String::from("stat");
    let s2 = String::from("22");
    let s3 = String::from("mortal");
    let args: Vec<_> = vec![s1, s2, s3];
    let dir = args.get(1).context(USAGE)?;
    let player_name = args.get(2).context(USAGE)?;

    // for entry in glob("22/**/*.json.gz").unwrap() {
    //     match entry {
    //         Ok(path) => println!("{:?}", path.display()),
    //
    //         // if the path matched but was unreadable,
    //         // thereby preventing its contents from matching
    //         Err(e) => println!("{:?}", e),
    //     }
    // }

    // let statx = glob(&format!("{dir}/**/*.json.gz")).unwrap();
    // for i in statx {
    //     println!("{:?}", i);
    // }

    let stat = Stat::from_dir(dir, player_name, false)?;
    println!("{stat}");

    Ok(())
}
