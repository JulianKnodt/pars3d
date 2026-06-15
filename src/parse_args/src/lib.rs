/// Parses arguments from a command line application.
/// Used for building CLI applications for pars3d.
#[macro_export]
macro_rules! parse_args {
  (
    $desc: expr,
  $( $StateName: ident ( $($flags: expr),+ ; $help: expr $(; $check: expr)?) => $field: ident : $t: ty = $def: expr $( => $auto:expr )?, )+) => {{
    #[derive(Debug)]
    struct Args {
      $(pub $field: $t,)+
    }
    impl Default for Args {
      fn default() -> Self {
        Self {
          $($field: $def,)+
        }
      }
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    pub enum State {
      Empty,
      $($StateName,)+
    }

    // For use in binaries
    #[allow(non_local_definitions)]
    #[macro_export]
    macro_rules! help {
      ($err: tt) => {{
        eprintln!("[ERROR]: {}", format!($err));
        help!();
      }};
      () => {{
        eprintln!($desc);
        $(
          let mut msg = String::from("\t");
          $(
            msg.push_str($flags);
            msg.push_str(" ");
          )+
          msg.push_str(": ");
          msg.push_str($help);
          msg.push_str(&format!(" = `{}`", $def));
          $( msg.push_str(&format!(" [No Value => {}]", $auto)); )?
          $(
            let mut raw = stringify!($check);
            if raw.get(0..1) == Some("|") && let Some(snd) = raw[1..].find('|') {
              raw = &raw[snd+2..];
            }
            msg.push_str(&format!(" [Must satisfy: {}]", raw.trim()));
          )?
          eprintln!("{msg}");
        )+
        return Ok(());
      }}
    }

    /*
    $(
      $(
        const _ : () = {
          assert_ne!($flags, "-h", "Cannot use `-h` for user flags since it conflicts with help");
        };
      )+
    )+
    */

    let mut args = Args::default();
    let mut state = State::Empty;

    for v in std::env::args().skip(1) {
      match v.as_str() {
        "UNMATCHABLE" => help!("???"),
        $($($flags |)+ "UNMATCHABLE" => {
          if state != State::Empty {
            help!("Expected {state:?}");
          }
          // this `if true` is here to prevent compiler complaints
          $(if true {
            args.$field = $auto;
            continue;
          })?
          state = State::$StateName;
          continue;
        })+
        "-h" | "--help" => help!(),
        v if v.starts_with("-") => help!("Unknown flag {v}"),
        _ => {}
      }

      match state {
        $(State::$StateName => {
          args.$field = match v.parse::<$t>() {
            Ok(s) => s,
            Err(e) => help!("Failed to parse ({v:?}), err {e:?}"),
          };
          state = State::Empty;
        })+
        State::Empty => help!("No positional arguments supported, got {v}"),
      }
    }
    let mut any_failed = false;

    $(
      #[allow(unused)]
      let failed = false;
      $( let failed = !$check(&args.$field); )?

      if failed {
        let mut msg = String::from("");
        $(
          msg.push_str($flags);
          msg.push_str("/");
        )+
        eprintln!("[ERROR]: `{}` is invalid for {}.", args.$field, msg.trim_end_matches("/"));
      }
      any_failed = failed || any_failed;
    )+

    if any_failed {
      help!();
    }

    if state != State::Empty {
      help!("Expecting parameter for {state:?}, but did not get any value");
    }

    args
  }}
}
