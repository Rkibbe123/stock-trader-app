$dir1 = "C:\github-personal\stock-trader-app"
$dir2 = "C:\git-personal\stock-trader-app"

$left = Get-ChildItem $dir1 -Recurse -File | Select-Object @{
  Name='RelPath'; Expression={ $_.FullName.Substring($dir1.Length).TrimStart('\') }
}

$right = Get-ChildItem $dir2 -Recurse -File | Select-Object @{
  Name='RelPath'; Expression={ $_.FullName.Substring($dir2.Length).TrimStart('\') }
}

Compare-Object $left $right -Property RelPath |
  ForEach-Object { $_.RelPath } |
  Sort-Object -Unique